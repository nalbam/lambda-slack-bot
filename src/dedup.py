"""DynamoDB-backed idempotency and thread conversation memory.

Single table, partition key `id`. Two key prefixes share the table:
- `dedup:{client_msg_id}` — one-shot reservation for request deduplication.
- `ctx:{thread_ts}` — conversation history for thread memory.

A GSI on `user` + `expire_at` enables per-user active-request counting
for throttling.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


@dataclass
class _BaseStore:
    table_name: str
    region: str
    _table: Any = None

    def _get_table(self) -> Any:
        if self._table is None:
            self._table = boto3.resource("dynamodb", region_name=self.region).Table(self.table_name)
        return self._table


class DedupStore(_BaseStore):
    """Atomic reservation for Slack retry deduplication + user throttle count."""

    GSI_NAME = "user-index"

    def reserve(self, event_key: str, user: str = "system", ttl_seconds: int = 3600) -> bool:
        """Return True if reservation succeeds, False if already reserved.

        Uses ConditionExpression=attribute_not_exists(id) for atomicity — no
        get-then-put race window.
        """
        table = self._get_table()
        expire_at = int(time.time()) + ttl_seconds
        try:
            table.put_item(
                Item={"id": f"dedup:{event_key}", "user": user, "expire_at": expire_at},
                ConditionExpression="attribute_not_exists(id)",
            )
            return True
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
                return False
            logger.warning("dedup reserve failed: %s", exc)
            raise

    def count_user_active(self, user: str) -> int:
        """Number of non-expired reservations for a user (throttle check)."""
        if not user:
            return 0
        table = self._get_table()
        now = int(time.time())
        try:
            res = table.query(
                IndexName=self.GSI_NAME,
                KeyConditionExpression="#u = :u AND expire_at > :now",
                ExpressionAttributeNames={"#u": "user"},
                ExpressionAttributeValues={":u": user, ":now": now},
                Select="COUNT",
            )
            return int(res.get("Count", 0))
        except ClientError as exc:
            logger.warning("count_user_active failed: %s", exc)
            return 0


class ConversationStore(_BaseStore):
    """Thread conversation history with TTL."""

    def get(self, thread_ts: str) -> list[dict[str, Any]]:
        if not thread_ts:
            return []
        table = self._get_table()
        try:
            res = table.get_item(Key={"id": f"ctx:{thread_ts}"})
        except ClientError as exc:
            logger.warning("conversation get failed: %s", exc)
            return []
        item = res.get("Item")
        if not item:
            return []
        raw = item.get("conversation") or "[]"
        try:
            messages = json.loads(raw)
            return messages if isinstance(messages, list) else []
        except json.JSONDecodeError:
            logger.warning("malformed conversation blob for %s", thread_ts)
            return []

    def put(
        self,
        thread_ts: str,
        user: str,
        messages: list[dict[str, Any]],
        ttl_seconds: int = 3600,
        max_chars: int = 4000,
    ) -> None:
        if not thread_ts:
            return
        trimmed = self.truncate_to_chars(messages, max_chars)
        table = self._get_table()
        try:
            table.put_item(
                Item={
                    "id": f"ctx:{thread_ts}",
                    "user": user or "unknown",
                    "expire_at": int(time.time()) + ttl_seconds,
                    "conversation": json.dumps(trimmed, ensure_ascii=False),
                }
            )
        except ClientError as exc:
            logger.warning("conversation put failed: %s", exc)

    @staticmethod
    def truncate_to_chars(messages: list[dict[str, Any]], max_chars: int) -> list[dict[str, Any]]:
        """Drop oldest messages until total serialized size <= max_chars."""
        if not messages:
            return []
        out = list(messages)
        while out and len(json.dumps(out, ensure_ascii=False)) > max_chars:
            out.pop(0)
        return out
