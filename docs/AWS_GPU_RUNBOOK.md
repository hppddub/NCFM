# AWS GPU Runbook — From Zero to an Auditable FT-NCFM Smoke Run

This runbook provisions a single EC2 host for the FT-NCFM replication. It does
not claim to reproduce the paper's VLA results: the current executable is still a
feature-level MNIST proxy, and the independent audit in `AUDIT_2026-08-15.md`
lists the scientific blockers.

## 1. Choose the hardware deliberately

| EC2 type | Accelerator | Recommended use here |
| --- | --- | --- |
| `g6.xlarge` | 1 NVIDIA L4, 24 GB | First infrastructure and CUDA smoke test |
| `g6e.xlarge` | 1 NVIDIA L40S, 48 GB | Larger single-GPU proxy work |
| `p5.4xlarge` | 1 NVIDIA H100, 80 GB | Closest AWS single-GPU memory shape; Region availability is limited |
| `p4d.24xlarge` | 8 NVIDIA A100, 40 GB each | Exact A100 architecture, but far larger than this proof of concept |
| `p4de.24xlarge` | 8 NVIDIA A100, 80 GB each | A100 80 GB, but eight GPUs and correspondingly expensive |

The paper reports one A100-SXM4-80GB. EC2 does not offer that exact one-GPU A100
shape. Start on `g6.xlarge`, verify the complete process, and only then decide
whether an eight-GPU P4 host is scientifically justified. The preflight script
queries current availability, account quota, and On-Demand pricing from AWS; no
price is hard-coded in this repository.

## 2. Secure the AWS account and create a budget

1. Sign in to the AWS console and enable MFA for the root user.
2. Do not use root credentials for the experiment. Use an IAM Identity Center
   administrator or the scoped deployment user below. The initial CloudFormation deployment needs
   permission to create EC2, VPC, IAM, S3, Lambda, EventBridge, CloudWatch Logs,
   and Systems Manager resources.
3. In **Billing and Cost Management → Budgets**, create a monthly cost budget with
   actual and forecast alerts to your email. Choose a limit below the unused AWS
   credits you are willing to spend on this experiment.

Budget alerts are delayed billing signals, not hard caps. The instance therefore
also has a systemd stop timer and a separate EventBridge/Lambda stop backstop.

If enabling AWS Organizations would expire free-plan credits, preserve them and
create the standalone-account deployment user instead:

```bash
bash infra/aws/cloudshell/create_deployer.sh us-east-2 --dry-run
bash infra/aws/cloudshell/create_deployer.sh us-east-2 --apply
```

The script creates no password or access key. In IAM, configure console access
privately, require a password reset, and enroll MFA before leaving the root
session. Its policies restrict CloudFormation to `ft-ncfm-gpu`, restrict EC2
launches to tagged `g6.xlarge` instances in the chosen Region, and require the
checked-in workload permissions boundary on every role it can create.
The operations policy grants only the five CloudShell actions needed to launch
the browser shell and forward the signed-in user's temporary credentials; it
does not grant `cloudshell:*`.

## 3. Open AWS CloudShell

CloudShell is the recommended beginner path: its AWS CLI is already installed and
uses the console session, so no long-lived access keys are copied to this PC.

1. Select the intended AWS Region in the console. For Toronto, begin with
   `ca-central-1`.
2. Select the CloudShell icon in the console header.
3. In CloudShell, clone the pushed replication branch:

```bash
git clone --branch codex/ft-ncfm-replication --single-branch \
  https://github.com/hppddub/NCFM.git ft-ncfm
cd ft-ncfm
```

Optional Windows-local route: install the current-user AWS CLI v2 MSI, run
`aws configure sso --profile ft-ncfm`, then append `--profile ft-ncfm` to AWS CLI
commands. Never paste access-key secrets into this repository or a chat.

## 4. Run the read-only preflight

The following checks a low-cost L4 in Canada Central with a three-hour runtime
guard. It does not create resources:

```bash
bash infra/aws/cloudshell/preflight.sh ca-central-1 g6.xlarge 180
```

Record the `PREFLIGHT_AZ` value. Confirm all of the following before continuing:

- the displayed AWS account is the intended credit-bearing account;
- the instance type is offered in the Region;
- the applied G- or P-family quota meets the required vCPU count;
- the current hourly price and guard-window estimate fit the budget.

If quota fails, open **Service Quotas → Amazon EC2** in the selected Region and
request the reported vCPU amount for either **Running On-Demand G and VT
instances** or **Running On-Demand P instances**. AWS may take time to approve it.

## 5. Deploy support resources with compute disabled

Replace `ca-central-1a` with the exact `PREFLIGHT_AZ` value:

```bash
bash infra/aws/cloudshell/deploy.sh \
  ca-central-1 g6.xlarge ca-central-1a false 180 ft-ncfm-gpu
```

This first deployment creates the isolated VPC, no-ingress security group,
least-scope instance role, encrypted result bucket, and related support resources.
`LaunchInstance=false` is the template default, so no GPU instance exists yet.
Review the stack in CloudFormation before proceeding.

## 6. Launch only after the price and quota check

The deployment script refuses billable compute unless the task-specific approval
variable is explicitly set in the same CloudShell session:

```bash
export FT_NCFM_ALLOW_BILLABLE=yes
bash infra/aws/cloudshell/deploy.sh \
  ca-central-1 g6.xlarge ca-central-1a true 180 ft-ncfm-gpu
unset FT_NCFM_ALLOW_BILLABLE
```

This is the point at which EC2 charges begin. The initial bootstrap pulls the AWS
DLAMI, clones the exact branch, builds the pinned CUDA container, starts the
automatic stop timer, and registers with Systems Manager.

## 7. Connect without SSH

In **Systems Manager → Session Manager**, select the instance named
`ft-ncfm-gpu-runner` and start a browser shell. No key pair or inbound port 22 is
used. Wait for bootstrap completion:

```bash
sudo test -f /opt/ft-ncfm/bootstrap-complete && echo ready
sudo tail -n 100 /var/log/ft-ncfm-bootstrap.log
nvidia-smi
```

If it is ready, switch to the repository owner:

```bash
sudo -iu ubuntu
cd /opt/ft-ncfm/repository
```

## 8. Run one nested smoke seed first

This is the fast, explicitly labeled 0.25%-of-full-corpus plumbing test:

```bash
bash infra/aws/remote/run_proxy.sh \
  configs/ft_ncfm/minivla_nested_5pct_smoke.yaml 42
```

The script records `nvidia-smi`, git state, GPU/driver/CUDA/cuDNN, AWS instance
metadata, container provenance, peak memory, metrics, influence tensors, and the
synthetic proxy. It uploads the run to the stack's encrypted S3 bucket.

Only after this run passes should the true 5%-of-full-source configuration be
attempted:

```bash
bash infra/aws/remote/run_proxy.sh configs/ft_ncfm/minivla_5pct.yaml 42
```

That corrected configuration uses all 60,000 real MNIST training records and
creates 3,000 synthetic feature proxies. It is still not a VLA benchmark.

## 9. Stop immediately and retrieve artifacts

The remote script prints this command on completion:

```bash
sudo systemctl poweroff
```

An EBS-backed instance stops when that command runs. EC2 compute billing stops as
the instance enters `stopping`, but EBS storage continues to incur a small charge.
Back in CloudShell, remove the instance from the stack after verifying the S3
upload:

```bash
bash infra/aws/cloudshell/deploy.sh \
  ca-central-1 g6.xlarge ca-central-1a false 180 ft-ncfm-gpu
```

List and download results:

```bash
bucket=$(aws cloudformation describe-stacks \
  --region ca-central-1 --stack-name ft-ncfm-gpu \
  --query "Stacks[0].Outputs[?OutputKey=='ArtifactBucketName'].OutputValue" \
  --output text)
aws s3 ls "s3://$bucket/runs/"
aws s3 sync "s3://$bucket/runs/" ./aws-results/
```

The S3 bucket is intentionally retained if the stack is deleted. After results
are safely copied, empty and delete the bucket manually if it is no longer needed.

## 10. Result gate and second audit

Before escalating from the smoke test, require:

- the manifest says CUDA was available and names the expected GPU;
- all losses are finite and the declared convergence check passes;
- source, synthetic, and overall fractions are unambiguous;
- the git SHA and container/base-image provenance are present;
- S3 contains the raw metrics and tensors for the run;
- the EC2 instance is stopped or absent.

Then run a second independent audit over the AWS manifest, raw metrics, S3 file
inventory, cost window, and convergence comparison. Paper-level work remains
blocked on train/reference/test isolation, sign ablations, reusable `(V,L,A)`
coresets, simulator counterexamples, all six baselines, and downstream evaluation.

## Official references

- [AWS EC2 P4 instances](https://aws.amazon.com/ec2/instance-types/p4/)
- [AWS EC2 accelerated-computing specifications](https://docs.aws.amazon.com/ec2/latest/instancetypes/ac.html)
- [AWS EC2 instance quotas](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html)
- [Finding the current DLAMI ID](https://docs.aws.amazon.com/dlami/latest/devguide/find-dlami-id.html)
- [AWS Systems Manager Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html)
- [AWS CLI IAM Identity Center authentication](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sso.html)
- [AWS CloudShell getting started](https://docs.aws.amazon.com/cloudshell/latest/userguide/getting-started.html)
- [AWS Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-create.html)
