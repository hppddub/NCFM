#!/usr/bin/env bash
set -euo pipefail

config="${1:-configs/ft_ncfm/minivla_nested_5pct_smoke.yaml}"
seed="${2:-42}"
runtime_env="/opt/ft-ncfm/aws-runtime.env"

if [[ ! -f "$runtime_env" ]]; then
  echo "Missing $runtime_env; run this on the provisioned EC2 instance." >&2
  exit 2
fi

set -a
source "$runtime_env"
set +a

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

run_id="$(date -u +%Y%m%dT%H%M%SZ)-seed-${seed}"
output_dir="artifacts/aws/$run_id"
mkdir -p "$output_dir" data_cache

nvidia-smi | tee "$output_dir/nvidia-smi.txt"
git rev-parse HEAD >"$output_dir/git-sha.txt"
git status --short >"$output_dir/git-status.txt"

base_digest="$(docker image inspect pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime \
  --format '{{index .RepoDigests 0}}' 2>/dev/null || true)"

docker run --rm --gpus all --shm-size=16g \
  --env "FT_NCFM_AWS_REGION=$FT_NCFM_AWS_REGION" \
  --env "FT_NCFM_AWS_AMI_ID=$FT_NCFM_AWS_AMI_ID" \
  --env "FT_NCFM_AWS_AVAILABILITY_ZONE=$FT_NCFM_AWS_AVAILABILITY_ZONE" \
  --env "FT_NCFM_AWS_INSTANCE_ID=$FT_NCFM_AWS_INSTANCE_ID" \
  --env "FT_NCFM_AWS_INSTANCE_TYPE=$FT_NCFM_AWS_INSTANCE_TYPE" \
  --env "FT_NCFM_AWS_MAX_RUNTIME_MINUTES=$FT_NCFM_AWS_MAX_RUNTIME_MINUTES" \
  --env "FT_NCFM_AWS_STACK_NAME=$FT_NCFM_AWS_STACK_NAME" \
  --env "FT_NCFM_CONTAINER_IMAGE=$FT_NCFM_CONTAINER_IMAGE" \
  --env "FT_NCFM_BASE_IMAGE_DIGEST=$base_digest" \
  --volume "$repo_root/artifacts:/workspace/artifacts" \
  --volume "$repo_root/data_cache:/workspace/data_cache" \
  "$FT_NCFM_CONTAINER_IMAGE" \
  python -m ft_ncfm.experiment \
    --config "$config" \
    --seed "$seed" \
    --output-dir "$output_dir"

aws s3 sync "$output_dir" "s3://$FT_NCFM_ARTIFACT_BUCKET/runs/$run_id/" --only-show-errors
echo "Uploaded: s3://$FT_NCFM_ARTIFACT_BUCKET/runs/$run_id/"
echo "Stop now: sudo systemctl poweroff"
