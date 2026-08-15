# AWS GPU Infrastructure

Use `docs/AWS_GPU_RUNBOOK.md` for the complete procedure. The CloudFormation
template defaults to `LaunchInstance=false`; supporting resources can therefore be
created and inspected before any GPU compute begins.

The stack uses an AWS Deep Learning Base GPU AMI, Session Manager with no inbound
ports, an encrypted EBS root volume, a retained encrypted S3 artifact bucket, an
in-guest systemd stop timer, and an independent EventBridge/Lambda stop backstop.
