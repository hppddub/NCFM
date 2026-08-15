#!/usr/bin/env bash
set -euo pipefail

region="${1:-us-east-2}"
mode="${2:---dry-run}"
user_name="ft-ncfm-deployer"

if [[ "$mode" != "--dry-run" && "$mode" != "--apply" ]]; then
  echo "Usage: $0 [REGION] [--dry-run|--apply]" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
policy_root="$repo_root/infra/aws/iam"
account_id="$(aws sts get-caller-identity --query Account --output text)"
quota_code="$(aws service-quotas list-service-quotas \
  --service-code ec2 \
  --region "$region" \
  --query "Quotas[?QuotaName=='Running On-Demand G and VT instances'].QuotaCode | [0]" \
  --output text)"

if [[ -z "$quota_code" || "$quota_code" == "None" ]]; then
  echo "Could not resolve the G-family quota code in $region." >&2
  exit 3
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf -- "$tmp_dir"' EXIT

render_policy() {
  local source="$1"
  local destination="$2"
  sed \
    -e "s/@@ACCOUNT_ID@@/$account_id/g" \
    -e "s/@@REGION@@/$region/g" \
    -e "s/@@G_QUOTA_CODE@@/$quota_code/g" \
    "$source" >"$destination"
}

render_policy "$policy_root/workload-permissions-boundary.json" "$tmp_dir/boundary.json"
render_policy "$policy_root/deployer-control-policy.json" "$tmp_dir/control.json"
render_policy "$policy_root/deployer-resources-policy.json" "$tmp_dir/resources.json"
render_policy "$policy_root/deployer-operations-policy.json" "$tmp_dir/operations.json"

for policy in boundary control resources operations; do
  echo "Validating $policy policy"
  findings="$(aws accessanalyzer validate-policy \
    --policy-type IDENTITY_POLICY \
    --policy-document "file://$tmp_dir/$policy.json" \
    --query "findings[?findingType=='ERROR']" \
    --output json)"
  if [[ "$findings" != "[]" ]]; then
    echo "$findings" >&2
    exit 4
  fi
done

echo "Region: $region"
echo "User: $user_name"
echo "G-family quota code: $quota_code"
echo "Allowed launch type: g6.xlarge"
echo "Allowed stack: ft-ncfm-gpu"

if [[ "$mode" == "--dry-run" ]]; then
  echo "Dry run complete. No IAM resources were created."
  exit 0
fi

declare -A policy_files=(
  [FTNCFMWorkloadBoundary]="$tmp_dir/boundary.json"
  [FTNCFMDeployerControl]="$tmp_dir/control.json"
  [FTNCFMDeployerResources]="$tmp_dir/resources.json"
  [FTNCFMDeployerOperations]="$tmp_dir/operations.json"
)

for policy_name in "${!policy_files[@]}"; do
  policy_arn="arn:aws:iam::$account_id:policy/$policy_name"
  if aws iam get-policy --policy-arn "$policy_arn" >/dev/null 2>&1; then
    echo "Refusing to overwrite existing policy: $policy_arn" >&2
    exit 5
  fi
done

if aws iam get-user --user-name "$user_name" >/dev/null 2>&1; then
  echo "Refusing to overwrite existing user: $user_name" >&2
  exit 6
fi

for policy_name in FTNCFMWorkloadBoundary FTNCFMDeployerControl FTNCFMDeployerResources FTNCFMDeployerOperations; do
  aws iam create-policy \
    --policy-name "$policy_name" \
    --policy-document "file://${policy_files[$policy_name]}" \
    --tags Key=Project,Value=ft-ncfm >/dev/null
done

aws iam create-user \
  --user-name "$user_name" \
  --tags Key=Project,Value=ft-ncfm Key=Purpose,Value=GPUDeployment >/dev/null

for policy_name in FTNCFMDeployerControl FTNCFMDeployerResources FTNCFMDeployerOperations; do
  aws iam attach-user-policy \
    --user-name "$user_name" \
    --policy-arn "arn:aws:iam::$account_id:policy/$policy_name"
done

echo "Created $user_name without a console password or access key."
echo "Configure console access privately in IAM, require a password reset, and enable MFA."
