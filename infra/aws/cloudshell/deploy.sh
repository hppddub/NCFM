#!/usr/bin/env bash
set -euo pipefail

region="${1:-ca-central-1}"
instance_type="${2:-g6.xlarge}"
availability_zone="${3:-}"
launch_instance="${4:-false}"
max_runtime_minutes="${5:-180}"
stack_name="${6:-ft-ncfm-gpu}"

if [[ -z "$availability_zone" ]]; then
  echo "Usage: $0 REGION INSTANCE_TYPE AVAILABILITY_ZONE [true|false] [MAX_MINUTES] [STACK_NAME]" >&2
  exit 2
fi

if [[ "$launch_instance" == "true" && "${FT_NCFM_ALLOW_BILLABLE:-no}" != "yes" ]]; then
  echo "Billable launch blocked." >&2
  echo "After checking the live price and budget, run: export FT_NCFM_ALLOW_BILLABLE=yes" >&2
  exit 3
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
template="$repo_root/infra/aws/cloudformation/gpu-runner.yaml"

aws cloudformation deploy \
  --region "$region" \
  --stack-name "$stack_name" \
  --template-file "$template" \
  --capabilities CAPABILITY_IAM \
  --no-fail-on-empty-changeset \
  --parameter-overrides \
    "AvailabilityZone=$availability_zone" \
    "InstanceType=$instance_type" \
    "LaunchInstance=$launch_instance" \
    "MaxRuntimeMinutes=$max_runtime_minutes"

aws cloudformation describe-stacks \
  --region "$region" \
  --stack-name "$stack_name" \
  --query 'Stacks[0].Outputs' \
  --output table
