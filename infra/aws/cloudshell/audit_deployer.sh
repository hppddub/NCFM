#!/usr/bin/env bash
set -euo pipefail

region="${1:-us-east-2}"
user_name="ft-ncfm-deployer"
expected_account="${2:-}"
export AWS_PAGER=""

account_id="$(aws sts get-caller-identity --query Account --output text)"
if [[ -n "$expected_account" && "$account_id" != "$expected_account" ]]; then
  echo "Wrong AWS account: expected $expected_account, authenticated to $account_id." >&2
  exit 2
fi

user_arn="arn:aws:iam::$account_id:user/$user_name"
token_time="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
token_context="ContextKeyName=aws:TokenIssueTime,ContextKeyValues=$token_time,ContextKeyType=date"
checks=0

decision() {
  local action="$1"
  local resource="$2"
  shift 2
  local contexts=("$token_context" "$@")

  aws iam simulate-principal-policy \
    --policy-source-arn "$user_arn" \
    --action-names "$action" \
    --resource-arns "$resource" \
    --context-entries "${contexts[@]}" \
    --query 'EvaluationResults[0].EvalDecision' \
    --output text
}

expect_decision() {
  local expected="$1"
  local label="$2"
  local action="$3"
  local resource="$4"
  shift 4

  local actual
  actual="$(decision "$action" "$resource" "$@")"
  if [[ "$actual" != "$expected" ]]; then
    echo "FAIL: $label (expected $expected, got $actual)" >&2
    exit 3
  fi
  printf 'PASS: %-48s %s\n' "$label" "$actual"
  checks=$((checks + 1))
}

echo "Auditing $user_arn in $region"

attached="$(aws iam list-attached-user-policies \
  --user-name "$user_name" \
  --query 'sort(AttachedPolicies[].PolicyName)' \
  --output json)"
python3 - "$attached" <<'PY'
import json
import sys

actual = json.loads(sys.argv[1])
expected = [
    "FTNCFMDeployerControl",
    "FTNCFMDeployerOperations",
    "FTNCFMDeployerResources",
]
if actual != expected:
    raise SystemExit(f"Unexpected attached policies: {actual!r}")
PY
echo "PASS: exactly three scoped managed policies are attached"
checks=$((checks + 1))

access_keys="$(aws iam list-access-keys --user-name "$user_name" --query 'AccessKeyMetadata' --output json)"
[[ "$access_keys" == "[]" ]] || {
  echo "FAIL: $user_name has an access key." >&2
  exit 4
}
echo "PASS: no long-lived IAM access keys"
checks=$((checks + 1))

mfa_count="$(aws iam list-mfa-devices --user-name "$user_name" --query 'length(MFADevices)' --output text)"
[[ "$mfa_count" == "1" ]] || {
  echo "FAIL: expected one MFA device, found $mfa_count." >&2
  exit 5
}
echo "PASS: one MFA device is assigned"
checks=$((checks + 1))

role_arn="arn:aws:iam::$account_id:role/ft-ncfm/ft-ncfm-instance-role"
profile_arn="arn:aws:iam::$account_id:instance-profile/ft-ncfm/ft-ncfm-instance-profile"
boundary_arn="arn:aws:iam::$account_id:policy/FTNCFMWorkloadBoundary"

expect_decision allowed "create boundary-constrained workload role" \
  iam:CreateRole "$role_arn" \
  "ContextKeyName=iam:PermissionsBoundary,ContextKeyValues=$boundary_arn,ContextKeyType=string"
expect_decision implicitDeny "reject workload role without boundary" \
  iam:CreateRole "$role_arn"
expect_decision allowed "pass workload role only to EC2" \
  iam:PassRole "$role_arn" \
  "ContextKeyName=iam:PassedToService,ContextKeyValues=ec2.amazonaws.com,ContextKeyType=string"
expect_decision implicitDeny "reject passing workload role to CodeBuild" \
  iam:PassRole "$role_arn" \
  "ContextKeyName=iam:PassedToService,ContextKeyValues=codebuild.amazonaws.com,ContextKeyType=string"
expect_decision allowed "create exact project instance profile" \
  iam:CreateInstanceProfile "$profile_arn"

expect_decision allowed "create exact CloudFormation stack" \
  cloudformation:CreateStack \
  "arn:aws:cloudformation:$region:$account_id:stack/ft-ncfm-gpu/audit"
expect_decision implicitDeny "reject an unrelated CloudFormation stack" \
  cloudformation:CreateStack \
  "arn:aws:cloudformation:$region:$account_id:stack/unrelated/audit"

bucket_arn="arn:aws:s3:::ft-ncfm-artifacts-$account_id-$region"
expect_decision allowed "create exact artifact bucket" s3:CreateBucket "$bucket_arn"
expect_decision implicitDeny "reject an unrelated S3 bucket" \
  s3:CreateBucket "arn:aws:s3:::unrelated-$account_id-$region"

instance_arn="arn:aws:ec2:$region:$account_id:instance/i-audit"
volume_arn="arn:aws:ec2:$region:$account_id:volume/vol-audit"
eni_arn="arn:aws:ec2:$region:$account_id:network-interface/eni-audit"
subnet_arn="arn:aws:ec2:$region:$account_id:subnet/subnet-audit"
sg_arn="arn:aws:ec2:$region:$account_id:security-group/sg-audit"
image_arn="arn:aws:ec2:$region::image/ami-audit"
snapshot_arn="arn:aws:ec2:$region::snapshot/snap-audit"

expect_decision allowed "launch tagged g6.xlarge instance leg" \
  ec2:RunInstances "$instance_arn" \
  "ContextKeyName=aws:RequestTag/Project,ContextKeyValues=ft-ncfm,ContextKeyType=string" \
  "ContextKeyName=ec2:InstanceType,ContextKeyValues=g6.xlarge,ContextKeyType=string"
expect_decision implicitDeny "reject expensive p4d instance leg" \
  ec2:RunInstances "$instance_arn" \
  "ContextKeyName=aws:RequestTag/Project,ContextKeyValues=ft-ncfm,ContextKeyType=string" \
  "ContextKeyName=ec2:InstanceType,ContextKeyValues=p4d.24xlarge,ContextKeyType=string"
expect_decision allowed "launch encrypted 200-GiB volume leg" \
  ec2:RunInstances "$volume_arn" \
  "ContextKeyName=ec2:Encrypted,ContextKeyValues=true,ContextKeyType=boolean" \
  "ContextKeyName=ec2:VolumeSize,ContextKeyValues=200,ContextKeyType=numeric"
expect_decision implicitDeny "reject 201-GiB volume leg" \
  ec2:RunInstances "$volume_arn" \
  "ContextKeyName=ec2:Encrypted,ContextKeyValues=true,ContextKeyType=boolean" \
  "ContextKeyName=ec2:VolumeSize,ContextKeyValues=201,ContextKeyType=numeric"
expect_decision allowed "authorize required network-interface leg" ec2:RunInstances "$eni_arn"
expect_decision allowed "use Project-tagged subnet leg" \
  ec2:RunInstances "$subnet_arn" \
  "ContextKeyName=ec2:ResourceTag/Project,ContextKeyValues=ft-ncfm,ContextKeyType=string"
expect_decision implicitDeny "reject unrelated subnet leg" \
  ec2:RunInstances "$subnet_arn" \
  "ContextKeyName=ec2:ResourceTag/Project,ContextKeyValues=other,ContextKeyType=string"
expect_decision allowed "use Project-tagged security-group leg" \
  ec2:RunInstances "$sg_arn" \
  "ContextKeyName=ec2:ResourceTag/Project,ContextKeyValues=ft-ncfm,ContextKeyType=string"
expect_decision allowed "authorize public DLAMI image leg" ec2:RunInstances "$image_arn"
expect_decision allowed "authorize public DLAMI snapshot leg" ec2:RunInstances "$snapshot_arn"

guard_role_arn="arn:aws:iam::$account_id:role/ft-ncfm/ft-ncfm-guard-function-role"
expect_decision allowed "pass guard role only to Lambda" \
  iam:PassRole "$guard_role_arn" \
  "ContextKeyName=iam:PassedToService,ContextKeyValues=lambda.amazonaws.com,ContextKeyType=string"
expect_decision allowed "create exact runtime guard function" \
  lambda:CreateFunction "arn:aws:lambda:$region:$account_id:function:ft-ncfm-runtime-guard"
expect_decision implicitDeny "reject unrelated Lambda function" \
  lambda:CreateFunction "arn:aws:lambda:$region:$account_id:function:unrelated"
expect_decision allowed "create exact runtime guard schedule" \
  events:PutRule "arn:aws:events:$region:$account_id:rule/ft-ncfm-runtime-guard"
expect_decision implicitDeny "reject unrelated EventBridge rule" \
  events:PutRule "arn:aws:events:$region:$account_id:rule/unrelated"

quota_code="$(aws service-quotas list-service-quotas \
  --service-code ec2 \
  --region "$region" \
  --query "Quotas[?QuotaName=='Running On-Demand G and VT instances'].QuotaCode | [0]" \
  --output text)"
expect_decision allowed "request only G-family quota" \
  servicequotas:RequestServiceQuotaIncrease \
  "arn:aws:servicequotas:$region:$account_id:ec2/$quota_code"

echo
echo "AUDIT PASSED: $checks least-privilege checks"
