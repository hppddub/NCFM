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


def test_deploy_script_requires_explicit_billable_acknowledgement() -> None:
    script = (REPOSITORY_ROOT / "infra/aws/cloudshell/deploy.sh").read_text(encoding="utf-8")

    assert "FT_NCFM_ALLOW_BILLABLE" in script
    assert "Billable launch blocked" in script
    assert 'launch_instance="${4:-false}"' in script
