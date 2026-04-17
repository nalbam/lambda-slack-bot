# lambda-slack-bot

Slack 멘션을 AWS Lambda에서 처리하고, OpenAI 또는 AWS Bedrock LLM을 선택해서 의도 파악 → 도구 실행(반복) → 결과 합성 응답을 수행하는 봇입니다.

## 주요 기능

- Slack `app_mention` 이벤트 처리 (`slack-bolt` + Lambda 핸들러)
- LLM 3단계 흐름
  - 의도/계획 수립
  - Tool 호출 반복 실행
  - 최종 응답 합성
- Tool 제공
  - 첨부 이미지 읽기
  - 스레드 히스토리 조회
  - Slack 메시지 검색
  - 웹 검색
  - 이미지 생성 후 Slack 업로드
- 환경 변수로 텍스트/이미지 모델 선택

## 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|------|------|--------|------|
| `SLACK_BOT_TOKEN` | ✅ | — | Slack Bot User OAuth Token (`xoxb-…`) |
| `SLACK_SIGNING_SECRET` | ✅ | — | Slack 앱 Signing Secret |
| `OPENAI_API_KEY` | ✅ (OpenAI 사용 시) | — | OpenAI API 키 |
| `LLM_PROVIDER` | | `openai` | `openai` 또는 `bedrock` |
| `LLM_MODEL` | | `gpt-4.1-mini` | 텍스트 모델 |
| `IMAGE_PROVIDER` | | `openai` | `openai` 또는 `bedrock` |
| `IMAGE_MODEL` | | `gpt-image-1` | 이미지 생성 모델 |
| `AGENT_MAX_STEPS` | | `3` | 에이전트 최대 반복 횟수 |
| `RESPONSE_LANGUAGE` | | `ko` | 응답 언어 (`ko` / `en`) |

## 로컬 테스트

### 1. 환경 설정

```bash
# 의존 패키지 설치
pip install -r requirements.txt

# 환경 변수 파일 생성
cp .env.example .env.local
```

`.env.local` 을 열고 실제 값으로 채웁니다.  
최소 설정: `OPENAI_API_KEY` (OpenAI 사용 시)

```dotenv
OPENAI_API_KEY=sk-...
SLACK_BOT_TOKEN=xoxb-...        # Slack 도구(검색, 업로드 등)를 쓰려면 필요
SLACK_SIGNING_SECRET=...
```

> `.env.local` 은 `.gitignore` 에 등록되어 있어 커밋되지 않습니다.

---

### 2. CLI 질문·응답 테스트 (Slack 불필요)

에이전트에 직접 텍스트를 넣고 응답을 확인합니다.  
Slack 연결 없이 LLM 응답, 웹 검색 등을 테스트할 수 있습니다.

```bash
# 인수로 질문 전달
python localtest.py "오늘 서울 날씨 알려줘"

# 대화형 입력 (Ctrl+D 로 종료)
python localtest.py
```

실행 예시:

```
▶ 질문: 오늘 서울 날씨 알려줘

처리 중...

────────────────────────────────────────────────────
오늘 서울의 날씨는 맑고 최고 기온은 약 22°C 입니다. ...
────────────────────────────────────────────────────
```

> **참고**: `SLACK_BOT_TOKEN` 이 설정되지 않은 경우 Slack 관련 도구(스레드 조회, 메시지 검색, 이미지 업로드)는 빈 결과를 반환합니다.

---

### 3. 단위 테스트

```bash
python -m pytest tests/ -v
```

## Lambda 엔트리포인트

- `app.lambda_handler`
