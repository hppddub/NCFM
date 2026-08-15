import json
from pathlib import Path

import pytest
import yaml

from ft_ncfm.config import load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class CloudFormationLoader(yaml.SafeLoader):
    pass


def _construct_cloudformation_tag(loader, tag_suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_mapping(node)


CloudFormationLoader.add_multi_constructor("!", _construct_cloudformation_tag)


def test_ratio_configs_distinguish_nested_smoke_from_true_five_percent() -> None:
    smoke = load_config(REPOSITORY_ROOT / "configs/ft_ncfm/minivla_nested_5pct_smoke.yaml")
    true_five = load_config(REPOSITORY_ROOT / "configs/ft_ncfm/minivla_5pct.yaml")

    smoke_overall = smoke["data"]["train_ratio"] * smoke["distillation"]["coreset_ratio"]
    true_overall = true_five["data"]["train_ratio"] * true_five["distillation"]["coreset_ratio"]

    assert smoke_overall == pytest.approx(0.0025)
    assert true_overall == pytest.approx(0.05)
    assert smoke["runtime"]["deterministic"] is True
    assert true_five["runtime"]["deterministic"] is True
    assert smoke["runtime"]["deterministic_warn_only"] is False
    assert true_five["runtime"]["deterministic_warn_only"] is False


def test_cloudformation_defaults_to_no_compute_and_no_inbound_access() -> None:
    template_path = REPOSITORY_ROOT / "infra/aws/cloudformation/gpu-runner.yaml"
    template = yaml.load(template_path.read_text(encoding="utf-8"), Loader=CloudFormationLoader)

    assert template["Parameters"]["LaunchInstance"]["Default"] == "false"
    resources = template["Resources"]
    security_group = resources["GpuSecurityGroup"]["Properties"]
    assert "SecurityGroupIngress" not in security_group

    instance = resources["GpuInstance"]
    assert instance["Condition"] == "LaunchCompute"
    assert instance["Properties"]["MetadataOptions"]["HttpTokens"] == "required"
    assert instance["Properties"]["InstanceInitiatedShutdownBehavior"] == "stop"
    assert resources["GuardFunction"]["Condition"] == "LaunchCompute"
    assert resources["GuardSchedule"]["Condition"] == "LaunchCompute"
    assert resources["ArtifactBucket"]["DeletionPolicy"] == "Retain"
    assert (
        resources["InstanceRole"]["Properties"]["PermissionsBoundary"]
        == "WorkloadPermissionsBoundaryArn"
    )
    assert (
        resources["GuardFunctionRole"]["Properties"]["PermissionsBoundary"]
        == "WorkloadPermissionsBoundaryArn"
    )


def test_deploy_script_requires_explicit_billable_acknowledgement() -> None:
    script = (REPOSITORY_ROOT / "infra/aws/cloudshell/deploy.sh").read_text(encoding="utf-8")

    assert "FT_NCFM_ALLOW_BILLABLE" in script
    assert "Billable launch blocked" in script
    assert 'launch_instance="${4:-false}"' in script
    assert "CAPABILITY_NAMED_IAM" in script
    assert "WorkloadPermissionsBoundaryArn=$boundary_arn" in script


def test_preflight_uses_json_for_paginated_quota_query() -> None:
    script = (REPOSITORY_ROOT / "infra/aws/cloudshell/preflight.sh").read_text(encoding="utf-8")

    assert "Quotas[?QuotaName=='$quota_name'].Value | [0]" in script
    assert "--output json" in script
    assert 'quota_pass="false"' in script


def test_deployer_policy_is_low_cost_and_stack_scoped() -> None:
    policy_path = REPOSITORY_ROOT / "infra/aws/iam/deployer-resources-policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    encoded = json.dumps(policy)

    assert '"ec2:InstanceType": "g6.xlarge"' in encoded
    assert "p4d.24xlarge" not in encoded
    assert "p5.4xlarge" not in encoded

    control_path = REPOSITORY_ROOT / "infra/aws/iam/deployer-control-policy.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    control_encoded = json.dumps(control)
    assert "stack/ft-ncfm-gpu/*" in control_encoded
    assert "iam:PermissionsBoundary" in control_encoded

    operations_path = REPOSITORY_ROOT / "infra/aws/iam/deployer-operations-policy.json"
    operations = json.loads(operations_path.read_text(encoding="utf-8"))
    operations_encoded = json.dumps(operations)
    assert "cloudshell:PutCredentials" in operations_encoded
    assert "cloudshell:*" not in operations_encoded
