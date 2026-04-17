# lambda-slack-bot

Slack 멘션·DM 을 AWS Lambda 에서 처리하고, OpenAI 또는 AWS Bedrock LLM 으로 네이티브 **function calling** 기반 툴 오케스트레이션을 수행하는 봇입니다. `lambda-gurumi-ai-bot`·`lambda-slack-ai-bot` 의 대체를 목표로 합니다.

## 주요 기능

- **이벤트**: `app_mention`, DM(`message.im`)
- **Provider**: OpenAI · Bedrock(Anthropic Claude 3/3.5 · Amazon Nova) 선택 가능
- **Tools (네이티브 function calling)**
  - `read_attached_images` — 첨부 이미지 Vision 요약
  - `fetch_thread_history` — 스레드 히스토리 조회
  - `search_slack_messages` — Slack 메시지 검색
  - `search_web` — Tavily (TAVILY_API_KEY 설정 시) 또는 DuckDuckGo
  - `generate_image` — 이미지 생성 후 Slack 업로드
- **Production 기반**
  - DynamoDB 조건부 put 으로 Slack 재시도 **중복 제거**
  - 채널 allowlist · 유저당 동시 요청 **throttle**
  - DynamoDB 기반 **스레드 대화 메모리** (TTL 1h)
  - 긴 응답 **3단계 분할** 전송 (코드블록 → 문단 → 문장)
  - 스트리밍 `chat_update`, `assistant_threads_setStatus` 타이핑 인디케이터
  - 구조화 JSON 로깅 + request_id, agent 루프 관찰값 기록
  - 에러 메시지 sanitize (토큰·경로 redaction)

## 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `SLACK_BOT_TOKEN` | ✅ | — | `xoxb-…` |
| `SLACK_SIGNING_SECRET` | ✅ | — | Slack Signing Secret |
| `OPENAI_API_KEY` | OpenAI 사용 시 | — | OpenAI API 키 |
| `TAVILY_API_KEY` | | — | 설정 시 Tavily 웹 검색 활성화 |
| `LLM_PROVIDER` | | `openai` | `openai` / `bedrock` |
| `LLM_MODEL` | | `gpt-4o-mini` | 텍스트 모델 |
| `IMAGE_PROVIDER` | | `openai` | `openai` / `bedrock` |
| `IMAGE_MODEL` | | `gpt-image-1` | 이미지 모델 |
| `AGENT_MAX_STEPS` | | `3` | tool 루프 최대 iteration |
| `RESPONSE_LANGUAGE` | | `ko` | `ko` / `en` |
| `DYNAMODB_TABLE_NAME` | | `lambda-slack-bot-dev` | dedup / 대화 저장 테이블 |
| `AWS_REGION` | | `us-east-1` | AWS 리전 |
| `ALLOWED_CHANNEL_IDS` | | (empty) | 콤마 구분. 비어있으면 모든 채널 허용 |
| `ALLOWED_CHANNEL_MESSAGE` | | — | 비허용 채널 응답 메시지 |
| `MAX_LEN_SLACK` | | `3000` | 메시지 분할 기준 (≥500) |
| `MAX_THROTTLE_COUNT` | | `100` | 유저별 동시 요청 상한 |
| `MAX_HISTORY_CHARS` | | `4000` | 저장되는 대화 직렬화 최대 길이 |
| `BOT_CURSOR` | | `:robot_face:` | 플레이스홀더·스트림 인디케이터 이모지 |
| `SYSTEM_MESSAGE` | | — | 시스템 프롬프트 오버라이드 |
| `LOG_LEVEL` | | `INFO` | 로그 레벨 |

## 모델 매트릭스

| 용도 | OpenAI | Bedrock |
|------|--------|---------|
| 텍스트 + tool calling | `gpt-4o-mini`, `gpt-4o` | `anthropic.claude-3-5-sonnet-...`, `amazon.nova-pro-v1:0` |
| 이미지 생성 | `gpt-image-1` | `amazon.titan-image-generator-v1`, `amazon.nova-canvas-v1:0`, `stability.stable-diffusion-xl-v1` |

Claude 는 Messages API (`tools=[{name, description, input_schema}]`), Nova 는 Converse API (`toolConfig`) 로 자동 분기됩니다.

## 로컬 개발

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt

cp .env.example .env.local      # 값 채우기

# CLI 실행
python localtest.py "오늘 서울 날씨"
python localtest.py --stream "React 훅 설명해줘"
python localtest.py              # 대화형

# 테스트
python -m pytest --cov=src --cov-report=term-missing
```

`.env.local` 은 `src/config.py` 가 python-dotenv 로 자동 로드합니다. `SLACK_BOT_TOKEN` 이 placeholder 이면 `localtest.py` 가 Slack 호출을 stub 으로 대체합니다.

## 배포 (Serverless Framework v3)

### 1. IAM OIDC role 준비 (한 번만)

`role/lambda-slack-bot` 을 AWS 계정에 생성하고 GitHub OIDC trust + 배포용 policy 를 연결합니다. 템플릿과 상세 절차는 `.github/aws-role/` 에 있습니다:

```bash
cd .github/aws-role
export NAME="lambda-slack-bot"
aws iam create-role --role-name "${NAME}" --assume-role-policy-document file://trust-policy.json
aws iam create-policy --policy-name "${NAME}" --policy-document file://role-policy.json
export ACCOUNT_ID=$(aws sts get-caller-identity | jq -r .Account)
aws iam attach-role-policy --role-name "${NAME}" --policy-arn "arn:aws:iam::${ACCOUNT_ID}:policy/${NAME}"
```

`trust-policy.json` 은 `nalbam/lambda-slack-bot` repo 의 OIDC 토큰을, `role-policy.json` 은 CloudFormation / Lambda / IAM / S3 / DynamoDB / API Gateway / CloudWatch Logs 권한을 (`lambda-slack-bot-*` 스코프) 포함합니다.

### 2. GitHub 저장소 설정

- **Secrets**: `AWS_ACCOUNT_ID`, `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, `OPENAI_API_KEY`, `TAVILY_API_KEY`(선택)
- **Variables**: `LLM_PROVIDER`, `LLM_MODEL`, `IMAGE_PROVIDER`, `IMAGE_MODEL`, `RESPONSE_LANGUAGE`, `ALLOWED_CHANNEL_IDS`, `ALLOWED_CHANNEL_MESSAGE`, `SYSTEM_MESSAGE`, `BOT_CURSOR`, `MAX_LEN_SLACK`, `MAX_THROTTLE_COUNT`, `MAX_HISTORY_CHARS`, `AGENT_MAX_STEPS`, `LOG_LEVEL`

### 3. 배포

`main` 브랜치에 push 하면 `.github/workflows/push-main.yml` 이 pytest → Serverless deploy 순으로 수행합니다. 수동 실행은 `workflow_dispatch`.

```bash
# 로컬 배포 (선택)
npm i -g serverless@3 && npm i serverless-python-requirements serverless-dotenv-plugin
serverless deploy --stage dev --region us-east-1
```

DynamoDB 테이블 (해시키 `id`, GSI `user-index`, TTL `expire_at`) 은 CloudFormation 이 생성합니다.

## 아키텍처

```
┌────────────────┐  POST /slack/events
│ Slack workspace│──────────────────┐
└────────────────┘                  ▼
                    ┌───────────────────────────────────┐
                    │ API Gateway → Lambda (app.py)     │
                    │ ├─ X-Slack-Retry-Num early return │
                    │ └─ SlackRequestHandler (Bolt)     │
                    └────────┬───────────────────┬──────┘
                             │                   │
                  ┌──────────▼─────────┐  ┌──────▼─────────┐
                  │ app_mention handler│  │ message handler│
                  └──────────┬─────────┘  └──────┬─────────┘
                             └──────┬────────────┘
                                    ▼
                ┌───────────────────────────────────────────┐
                │ _process()                                │
                │  1. DedupStore.reserve (conditional put)  │
                │  2. channel_allowed / throttle            │
                │  3. set_thread_status + placeholder say   │
                │  4. ConversationStore.get → history       │
                │  5. SlackMentionAgent.run ──┐             │
                │  6. send_long_message       │             │
                │  7. ConversationStore.put   │             │
                └─────────────────────────────┼─────────────┘
                                              │
                      ┌───────────────────────▼───────────────┐
                      │ Agent loop (native function calling)  │
                      │  LLM.chat(messages, tools=registry)   │
                      │   ↓ tool_calls?                       │
                      │  ToolExecutor.execute (per-call t/o)  │
                      │   ↓ role=tool result                  │
                      │  (loop up to AGENT_MAX_STEPS)         │
                      │  streaming chat_update on final step  │
                      └────────────┬──────────────────────────┘
                                   │
                   ┌───────────────┼────────────────┐
                   ▼               ▼                ▼
            ┌───────────┐   ┌────────────┐  ┌──────────────┐
            │ OpenAI    │   │ Bedrock    │  │ Slack Web API│
            │ Chat API  │   │ Messages / │  │ (tools)      │
            │ Vision    │   │ Converse   │  └──────────────┘
            └───────────┘   └────────────┘
                                   ▲
                                   │
                            ┌──────┴─────┐
                            │ DynamoDB   │
                            │ (dedup+ctx)│
                            └────────────┘
```

## Lambda 엔트리포인트

- `app.lambda_handler`

## Production 체크리스트 (수동 확인용)

- [ ] `@<bot>` 멘션에 정상 응답
- [ ] DM 대화에 응답
- [ ] 긴 응답이 여러 청크로 쪼개져 스레드에 전송됨
- [ ] 이미지 생성 요청 (`"고양이 그려줘"`) 이 업로드됨
- [ ] 이미지 첨부 요약 (`read_attached_images`) 동작
- [ ] `ALLOWED_CHANNEL_IDS` 외 채널에서 차단 메시지 표시
- [ ] Slack retry 중복 호출이 dedup 로 무시됨 (CloudWatch 로그에 `dedup.skip` 확인)
- [ ] `assistant_threads_setStatus` 타이핑 인디케이터 표시
- [ ] 같은 스레드 재멘션 시 이전 대화 맥락 참조

## 제외된 항목 (Phase 2 이상)

- Bedrock Knowledge Base (RAG) 통합
- `reaction_added` 이벤트 훅 및 도메인 특화 로직
- CloudWatch Alarms / X-Ray tracing
