#!/usr/bin/env bash
set -euo pipefail

region="${1:-us-east-2}"
stack_name="${2:-ft-ncfm-gpu}"
instance_type="${3:-g6.xlarge}"
root_volume_gib="${4:-200}"
max_runtime_minutes="${5:-180}"
export AWS_PAGER=""

if [[ "$instance_type" != "g6.xlarge" ]]; then
  echo "Dry-run blocked: the scoped deployer permits only g6.xlarge." >&2
  exit 2
fi

if (( root_volume_gib < 100 || root_volume_gib > 200 )); then
  echo "Dry-run blocked: root volume must be between 100 and 200 GiB." >&2
  exit 3
fi

launch_gate="$(aws cloudformation describe-stacks \
  --region "$region" \
  --stack-name "$stack_name" \
  --query "Stacks[0].Parameters[?ParameterKey=='LaunchInstance'].ParameterValue | [0]" \
  --output text)"
if [[ "$launch_gate" != "false" ]]; then
  echo "Dry-run blocked: $stack_name must first be deployed with LaunchInstance=false." >&2
  exit 4
fi

resource_id() {
  aws cloudformation describe-stack-resource \
    --region "$region" \
    --stack-name "$stack_name" \
    --logical-resource-id "$1" \
    --query 'StackResourceDetail.PhysicalResourceId' \
    --output text
}

subnet_id="$(resource_id PublicSubnet)"
security_group_id="$(resource_id GpuSecurityGroup)"
profile_name="$(resource_id InstanceProfile)"
ami_id="$(aws ssm get-parameter \
  --region "$region" \
  --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
  --query 'Parameter.Value' \
  --output text)"

tag_specs="$(python3 - "$max_runtime_minutes" <<'PY'
import json
import sys

minutes = sys.argv[1]
tags = [
    {"Key": "Name", "Value": "ft-ncfm-gpu-runner"},
    {"Key": "Project", "Value": "ft-ncfm"},
    {"Key": "FTNCFMMaxRuntimeMinutes", "Value": minutes},
]
print(json.dumps([
    {"ResourceType": "instance", "Tags": tags},
    {"ResourceType": "volume", "Tags": tags},
]))
PY
)"

block_devices="$(python3 - "$root_volume_gib" <<'PY'
import json
import sys

print(json.dumps([{
    "DeviceName": "/dev/sda1",
    "Ebs": {
        "DeleteOnTermination": True,
        "Encrypted": True,
        "VolumeSize": int(sys.argv[1]),
        "VolumeType": "gp3",
    },
}]))
PY
)"

network_interface="DeviceIndex=0,AssociatePublicIpAddress=true,DeleteOnTermination=true,SubnetId=$subnet_id,Groups=$security_group_id"

echo "Validating exact launch request without creating an instance"
printf '  AMI:              %s\n' "$ami_id"
printf '  Instance type:    %s\n' "$instance_type"
printf '  Subnet:           %s\n' "$subnet_id"
printf '  Security group:   %s\n' "$security_group_id"
printf '  Instance profile: %s\n' "$profile_name"
printf '  Root volume:      %s GiB encrypted gp3\n' "$root_volume_gib"

set +e
dry_run_output="$(aws ec2 run-instances \
  --region "$region" \
  --dry-run \
  --image-id "$ami_id" \
  --instance-type "$instance_type" \
  --count 1 \
  --iam-instance-profile "Name=$profile_name" \
  --network-interfaces "$network_interface" \
  --block-device-mappings "$block_devices" \
  --metadata-options HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=1 \
  --tag-specifications "$tag_specs" 2>&1)"
dry_run_status=$?
set -e

if [[ "$dry_run_status" -eq 255 && "$dry_run_output" == *"DryRunOperation"* ]]; then
  echo "DRY RUN PASSED: AWS authorized the exact request and created no instance."
  exit 0
fi

echo "DRY RUN FAILED:" >&2
echo "$dry_run_output" >&2
exit 5
