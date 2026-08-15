# AWS GPU Setup Status — 15 August 2026

This record contains no credentials, account identifiers, or live resource IDs.
The checks below were executed in `us-east-2` through the scoped
`ft-ncfm-deployer` IAM user.

## Current gate status

| Gate | Result |
| --- | --- |
| Scoped IAM identity | Pass |
| Independent least-privilege simulation | Pass, 28 checks |
| CloudFormation template validation | Pass |
| Support-only stack | `CREATE_COMPLETE` |
| Stack launch parameter | `LaunchInstance=false` |
| Exact EC2 `RunInstances --dry-run` | `DryRunOperation` (authorized) |
| Project EC2 instances | None |
| Project EBS volumes | None |
| Conditional Lambda/EventBridge resources | None |
| Security-group ingress rules | Zero |
| Artifact-bucket encryption | AES-256 |
| Artifact-bucket public-access blocks | All enabled |
| Workload-role permissions boundary | Attached |
| Monthly budget | USD 25; recorded spend USD 0 at verification time |
| G/VT On-Demand quota request | 4 vCPUs, denied by AWS; appeal deferred |
| Applied G/VT quota | 0 vCPUs; compute remains blocked |

## Selected smoke-test target

- Region: `us-east-2`
- Availability Zone selected by the latest preflight: `us-east-2b`
- Instance: `g6.xlarge` (NVIDIA L4, 4 vCPUs)
- Maximum runtime guard: 180 minutes
- Encrypted root volume: 200 GiB gp3
- Access: Systems Manager Session Manager; no inbound security-group rules
- Live Linux On-Demand query at verification time: USD 0.8048/hour
- Three-hour compute-only guard-window estimate: USD 2.41

The low-cost L4 run is an infrastructure and convergence smoke test. It is not an
exact reproduction of the paper's A100-SXM4-80GB environment.

## Current compute route

AWS compute remains disabled after the quota denial. The immediate experiment path
has moved to Colab Pro using the checked-in `COLAB_A100_RUNBOOK.md` and A100
notebook. The support-only AWS stack remains at `LaunchInstance=false` and no AWS
GPU launch is part of the Colab procedure.

## Future AWS gate

Do not enable AWS compute until an appeal is approved and the applied G/VT quota
is at least 4 vCPUs. When it is:

1. Rerun `infra/aws/cloudshell/preflight.sh` and record the current AZ and price.
2. Confirm the monthly budget and absence of existing project instances.
3. Deploy with `LaunchInstance=true` and the 180-minute guard.
4. Confirm the in-guest systemd stop timer and the independent EventBridge/Lambda
   backstop before starting the experiment.
5. Run the proxy convergence smoke test, upload manifests/results to the encrypted
   artifact bucket, stop the instance, and verify it is no longer running.
6. Analyze results and commission the planned second independent audit.

## Scientific scope

The first independent audit approved an inexpensive infrastructure smoke test but
not paper-comparable claims. Influence-sign ablations, true 5%-ratio accounting,
isolated train/reference/test splits, trainable `(V, L, A)` synthetic records,
simulator-grounded counterexamples, and the full baseline/evaluation block remain
research gates before a substantive A100 experiment.
