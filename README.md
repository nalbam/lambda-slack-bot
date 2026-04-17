# lambda-slack-bot

Slack 멘션·DM 을 AWS Lambda 에서 처리하고, OpenAI 또는 AWS Bedrock LLM 으로 네이티브 **function calling** 기반 툴 오케스트레이션을 수행하는 봇입니다. `lambda-gurumi-ai-bot`·`lambda-slack-ai-bot` 의 대체를 목표로 합니다.

## 봇의 처리 흐름 (절대 생략하지 않는다)

모든 사용자 메시지는 다음 네 단계를 **순서대로** 통과합니다:

```
질문 ── 의도·계획 ── 툴 사용 (반복) ── 응답
 (user)    (LLM)        (tools)        (LLM)
```

**의도 파악과 계획은 한 번의 LLM 호출로 통합**되어 있습니다 (OpenAI / Claude / Nova 의 native function calling). 같은 응답에 "무슨 요청인지 파악한 결과" 와 "다음에 부를 tool_calls" 가 함께 담겨 옵니다. 별도의 intent 분류 hop 을 추가하지 않습니다.

- **의도·계획은 LLM 이 한다.** 키워드 매칭(예: `"그려"` → 이미지)으로 우회하지 않는다. LLM 이 메시지를 읽고 `tool_calls` 로 의도를 표현한다.
- **단계 단축 금지.** 이미지 요청처럼 명확해 보여도 `LLM 판단 → generate_image tool → LLM 응답 합성` 전 과정을 거친다. 응답 합성 단계를 건너뛰면 caption·후속 대응·에러 처리가 사라진다.
- **Agent 루프는 `src/agent.py` 안에** 있고, `app.py` 는 Slack 관련 부분(placeholder, streaming, 히스토리) 만 담당한다.
- **속도 문제는 파이프라인 단축이 아닌** 스트리밍·비동기·모델 선택으로 해결한다.

## 주요 기능

- **이벤트**: `app_mention`, DM(`message.im`)
- **Provider**: OpenAI · AWS Bedrock(Anthropic Claude 3/3.5/4.x · Amazon Nova) · xAI(Grok) 선택 가능
- **Tools (네이티브 function calling)**
  - `read_attached_images` — 첨부 이미지 Vision 요약
  - `fetch_thread_history` — 스레드 히스토리 조회
  - `search_web` — Tavily (TAVILY_API_KEY 설정 시) 또는 DuckDuckGo
  - `generate_image` — 이미지 생성 후 Slack 업로드
- **Production 기반**
  - DynamoDB 조건부 put 으로 Slack 재시도 **중복 제거**
  - 채널 allowlist · 유저당 동시 요청 **throttle**
  - DynamoDB 기반 **스레드 대화 메모리** (TTL 1h)
  - 긴 응답 **계층적 분할** 전송 (코드블록 → 문단 → 문장 → hard slice), `chat.update` 가 `msg_too_long` 에 걸리지 않도록 `MAX_LEN_SLACK` 기반 rolling 스트리밍 + 최종 답변 자동 split
  - 스트리밍 `chat_postMessage` + 반복 `chat_update` fallback (네이티브 `chat.startStream`/`appendStream`/`stopStream` 은 AI 워크스페이스에서 추가 "searching" 상태 UI 를 띄워 두 개의 응답처럼 보이는 이슈 때문에 기본 비활성화, `enable_native=True` 로만 사용), `assistant_threads_setStatus` 타이핑 인디케이터
  - 구조화 JSON 로깅 + request_id, agent 루프 관찰값 기록
  - 에러 메시지 sanitize (토큰·경로 redaction)

## 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `SLACK_BOT_TOKEN` | ✅ | — | `xoxb-…` |
| `SLACK_SIGNING_SECRET` | ✅ | — | Slack Signing Secret |
| `OPENAI_API_KEY` | OpenAI 사용 시 | — | OpenAI API 키 |
| `XAI_API_KEY` | xAI 사용 시 | — | xAI (Grok) API 키 — https://console.x.ai |
| `TAVILY_API_KEY` | | — | 설정 시 Tavily 웹 검색 활성화 |
| `LLM_PROVIDER` | | `openai` | `openai` / `bedrock` / `xai` |
| `LLM_MODEL` | | `gpt-4o-mini` | 텍스트 모델 |
| `IMAGE_PROVIDER` | | `openai` | `openai` / `bedrock` / `xai` |
| `IMAGE_MODEL` | | `gpt-image-1` | 이미지 모델 |
| `AGENT_MAX_STEPS` | | `3` | tool 루프 최대 iteration |
| `RESPONSE_LANGUAGE` | | `ko` | `ko` / `en` |
| `DYNAMODB_TABLE_NAME` | | `lambda-slack-bot-dev` | dedup / 대화 저장 테이블 |
| `AWS_REGION` | | `us-east-1` | AWS 리전 |
| `ALLOWED_CHANNEL_IDS` | | (empty) | 콤마 구분. 비어있으면 모든 채널 허용 |
| `ALLOWED_CHANNEL_MESSAGE` | | — | 비허용 채널 응답 메시지 |
| `MAX_LEN_SLACK` | | `2000` | 메시지 분할 기준 (≥500). Slack `chat.update` 의 한계 회피용 안전 margin. |
| `MAX_OUTPUT_TOKENS` | | `4096` | LLM hop 당 출력 토큰 상한 (≥256) |
| `MAX_THROTTLE_COUNT` | | `100` | 유저별 동시 요청 상한 |
| `MAX_HISTORY_CHARS` | | `4000` | 저장되는 대화 직렬화 최대 길이 |
| `BOT_CURSOR` | | `:robot_face:` | 플레이스홀더·스트림 인디케이터 이모지 |
| `SYSTEM_MESSAGE` | | — | 시스템 프롬프트 오버라이드 |
| `LOG_LEVEL` | | `INFO` | 로그 레벨 |

## 모델 매트릭스

| 용도 | OpenAI | Bedrock | xAI (Grok) |
|------|--------|---------|------------|
| 텍스트 + tool calling | `gpt-4o-mini`, `gpt-4o`, `gpt-5-*`, `o1/o3/o4` | `us.anthropic.claude-opus-4-6-v1`, `us.anthropic.claude-sonnet-4-5-...`, `amazon.nova-pro-v1:0` | `grok-4-1-fast-reasoning`, `grok-4.20-0309-reasoning`, `grok-4.20-multi-agent-0309` |
| 이미지 생성 | `gpt-image-1`, `dall-e-3` | `amazon.nova-canvas-v1:0`, `amazon.titan-image-generator-v2:0` | `grok-imagine-image`, `grok-imagine-image-pro` |

- Claude 는 Messages API (`tools=[{name, description, input_schema}]`), Nova 는 Converse API (`toolConfig`) 로 자동 분기됩니다.
- xAI 는 OpenAI wire 호환이라 OpenAI Python SDK 에 `base_url="https://api.x.ai/v1"` 만 swap 해서 호출합니다. 별도 `XAIProvider` 클래스로 분리되어 있습니다.
- Bedrock 최신 모델은 `us./eu./apac./global.` inference-profile prefix 가 붙은 ID 로만 호출됩니다. `BedrockProvider` 가 자동 인식합니다.

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

- **Secrets**: `AWS_ACCOUNT_ID`, `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`, `OPENAI_API_KEY`, `XAI_API_KEY`(xAI 사용 시), `TAVILY_API_KEY`(선택)
- **Variables**: `LLM_PROVIDER`, `LLM_MODEL`, `IMAGE_PROVIDER`, `IMAGE_MODEL`, `RESPONSE_LANGUAGE`, `ALLOWED_CHANNEL_IDS`, `ALLOWED_CHANNEL_MESSAGE`, `SYSTEM_MESSAGE`, `BOT_CURSOR`, `MAX_LEN_SLACK`, `MAX_OUTPUT_TOKENS`, `MAX_THROTTLE_COUNT`, `MAX_HISTORY_CHARS`, `AGENT_MAX_STEPS`, `LOG_LEVEL`

### 3. 배포

`main` 브랜치에 push 하면 `.github/workflows/push-main.yml` 이 pytest → Serverless deploy 순으로 수행합니다. 수동 실행은 `workflow_dispatch`.

```bash
# 로컬 배포 (선택)
npm i -g serverless@3 && npm i serverless-python-requirements
# Secrets + Variables 를 현재 셸에 export 한 뒤
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
