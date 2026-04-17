# aws role

GitHub Actions 가 OIDC 로 AssumeRole 할 때 사용할 IAM role 과 policy 를 준비합니다. `push-main.yml` 워크플로우는 `role/lambda-slack-bot` 을 가정해 배포를 수행합니다.

```bash
export NAME="lambda-slack-bot"
```

## create role

```bash
export DESCRIPTION="${NAME} role"

aws iam create-role --role-name "${NAME}" --description "${DESCRIPTION}" --assume-role-policy-document file://trust-policy.json | jq .

aws iam get-role --role-name "${NAME}" | jq .
```

## create policy

```bash
export DESCRIPTION="${NAME} policy"

aws iam create-policy --policy-name "${NAME}" --policy-document file://role-policy.json | jq .

export ACCOUNT_ID=$(aws sts get-caller-identity | jq .Account -r)
export POLICY_ARN="arn:aws:iam::${ACCOUNT_ID}:policy/${NAME}"

aws iam get-policy --policy-arn "${POLICY_ARN}" | jq .

aws iam create-policy-version --policy-arn "${POLICY_ARN}" --policy-document file://role-policy.json --set-as-default | jq .
```

## attach role policy

```bash
aws iam attach-role-policy --role-name "${NAME}" --policy-arn "${POLICY_ARN}"
# aws iam attach-role-policy --role-name "${NAME}" --policy-arn "arn:aws:iam::aws:policy/PowerUserAccess"
# aws iam attach-role-policy --role-name "${NAME}" --policy-arn "arn:aws:iam::aws:policy/AdministratorAccess"
```

## add role-assume

```yaml

      - name: configure aws credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: "arn:aws:iam::${{ env.AWS_ACCOUNT_ID }}:role/lambda-slack-bot"
          role-session-name: github-actions-ci-bot
          aws-region: ${{ env.AWS_REGION }}

      - name: Sts GetCallerIdentity
        run: |
          aws sts get-caller-identity

```

## 권한 범위

- CloudFormation 스택 이름 `lambda-slack-bot-*`
- Lambda 함수 `lambda-slack-bot-*`
- IAM role `lambda-slack-bot-*` (Lambda 실행 role 생성/관리)
- S3 버킷 `lambda-slack-bot-*` (Serverless deployment bucket)
- DynamoDB 테이블 `lambda-slack-bot-*` (dedup/conversation 저장)
- API Gateway REST API
- CloudWatch Logs `/aws/lambda/lambda-slack-bot-*`

Bedrock KnowledgeBase / Agent / S3Vectors 권한은 현재 프로젝트 범위에서 제외되어 있습니다. RAG (Phase 2) 단계에서 확장 필요.
