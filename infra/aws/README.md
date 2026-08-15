# AWS GPU Infrastructure

Use `docs/AWS_GPU_RUNBOOK.md` for the complete procedure. The CloudFormation
template defaults to `LaunchInstance=false`; supporting resources can therefore be
created and inspected before any GPU compute begins.

The stack uses an AWS Deep Learning Base GPU AMI, Session Manager with no inbound
ports, an encrypted EBS root volume, a retained encrypted S3 artifact bucket, an
in-guest systemd stop timer, and an independent EventBridge/Lambda stop backstop.

For a standalone AWS account that must preserve free-plan credits, run
`infra/aws/cloudshell/create_deployer.sh REGION --dry-run` and review the policy
validation before repeating it with `--apply`. This path does not enable AWS
Organizations or IAM Identity Center. The deployment user is limited to the
`ft-ncfm-gpu` stack and `g6.xlarge`; workload roles must carry the checked-in
permissions boundary.

Before enabling billable compute, run the repeatable least-privilege audit as an
administrator and the exact EC2 authorization check as the deployer:

```bash
bash infra/aws/cloudshell/audit_deployer.sh us-east-2 797273592302
bash infra/aws/cloudshell/dry_run_launch.sh us-east-2 ft-ncfm-gpu
```

The second command always includes EC2's `--dry-run` flag. A successful result is
`DryRunOperation`; it does not create an instance.
