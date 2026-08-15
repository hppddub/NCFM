#!/usr/bin/env bash
set -euo pipefail

region="${1:-ca-central-1}"
instance_type="${2:-g6.xlarge}"
max_runtime_minutes="${3:-180}"

case "$instance_type" in
  p*) quota_name="Running On-Demand P instances" ;;
  g*) quota_name="Running On-Demand G and VT instances" ;;
  *) echo "Unsupported GPU family: $instance_type" >&2; exit 2 ;;
esac

echo "Authenticated identity"
aws sts get-caller-identity --output table

availability_zone="$({
  aws ec2 describe-instance-type-offerings \
    --region "$region" \
    --location-type availability-zone \
    --filters "Name=instance-type,Values=$instance_type" \
    --query 'InstanceTypeOfferings[0].Location' \
    --output text
} 2>/dev/null)"

if [[ -z "$availability_zone" || "$availability_zone" == "None" ]]; then
  echo "$instance_type is not offered in $region. Choose another Region or instance type." >&2
  exit 3
fi

required_vcpus="$(aws ec2 describe-instance-types \
  --region "$region" \
  --instance-types "$instance_type" \
  --query 'InstanceTypes[0].VCpuInfo.DefaultVCpus' \
  --output text)"

quota_value="$(aws service-quotas list-service-quotas \
  --service-code ec2 \
  --region "$region" \
  --query "Quotas[?QuotaName=='$quota_name']|[0].Value" \
  --output text)"

echo
echo "Preflight result"
printf '  Region:                 %s\n' "$region"
printf '  Availability Zone:      %s\n' "$availability_zone"
printf '  Instance type:          %s\n' "$instance_type"
printf '  Required vCPUs:         %s\n' "$required_vcpus"
printf '  Applied family quota:   %s\n' "$quota_value"
printf '  Runtime guard (minutes): %s\n' "$max_runtime_minutes"

if python3 - "$required_vcpus" "$quota_value" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[2]) >= float(sys.argv[1]) else 1)
PY
then
  echo "  Quota check:            PASS"
else
  echo "  Quota check:            FAIL"
  echo
  echo "Request at least $required_vcpus vCPUs for: $quota_name"
  echo "https://console.aws.amazon.com/servicequotas/home/services/ec2/quotas/?region=$region"
  exit 4
fi

echo
echo "Live On-Demand pricing (informational; verify in AWS Pricing Calculator)"
price_json="$(aws pricing get-products \
  --region us-east-1 \
  --service-code AmazonEC2 \
  --filters \
    "Type=TERM_MATCH,Field=instanceType,Value=$instance_type" \
    "Type=TERM_MATCH,Field=regionCode,Value=$region" \
    'Type=TERM_MATCH,Field=operatingSystem,Value=Linux' \
    'Type=TERM_MATCH,Field=tenancy,Value=Shared' \
    'Type=TERM_MATCH,Field=preInstalledSw,Value=NA' \
    'Type=TERM_MATCH,Field=capacitystatus,Value=Used' \
  --max-results 100 \
  --output json 2>/dev/null || true)"

python3 - "$max_runtime_minutes" "$price_json" <<'PY'
import json
import sys

minutes = int(sys.argv[1])
try:
    outer = json.loads(sys.argv[2])
    prices = []
    for encoded in outer.get("PriceList", []):
        offer = json.loads(encoded)
        for term in offer.get("terms", {}).get("OnDemand", {}).values():
            for dimension in term.get("priceDimensions", {}).values():
                if dimension.get("unit") == "Hrs":
                    value = float(dimension["pricePerUnit"]["USD"])
                    if value > 0:
                        prices.append(value)
    hourly = min(prices)
except (ValueError, KeyError):
    print("  Price query unavailable; use https://calculator.aws/")
else:
    print(f"  Linux On-Demand:        ${hourly:.4f}/hour")
    print(f"  Guard-window estimate:  ${hourly * minutes / 60:.2f} compute only")
PY

echo
echo "PREFLIGHT_AZ=$availability_zone"
