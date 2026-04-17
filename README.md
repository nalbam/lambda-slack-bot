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

- `SLACK_BOT_TOKEN`
- `SLACK_SIGNING_SECRET`
- `LLM_PROVIDER` (`openai` or `bedrock`)
- `LLM_MODEL` (예: `gpt-4.1-mini`, `anthropic.claude-3-5-sonnet-20240620-v1:0`)
- `IMAGE_PROVIDER` (`openai` or `bedrock`)
- `IMAGE_MODEL` (예: `gpt-image-1`, `amazon.titan-image-generator-v1`)
- `AGENT_MAX_STEPS` (기본: `3`)
- `RESPONSE_LANGUAGE` (기본: `ko`, 예: `en`)

## 로컬 테스트

```bash
python -m unittest discover -v
```

## Lambda 엔트리포인트

- `app.lambda_handler`
