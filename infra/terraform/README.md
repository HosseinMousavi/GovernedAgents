# Terraform: AWS ECS + RDS

```
cd infra/terraform
terraform init -backend-config="bucket=<state-bucket>" -backend-config="key=ssa.tfstate" -backend-config="region=us-east-1"
terraform apply -var db_password=... -var token_signing_key=... -var anthropic_api_key=... -var openai_api_key=...
```

What you get (all free-tier eligible or a few cents/hour):

| Resource | Size | Why |
|---|---|---|
| RDS Postgres 16 | db.t3.micro, 20 GB | pgvector + business tables |
| ECS Fargate x2 | 0.25 vCPU / 0.5 GB | `mcp` and `agents` services, one image |
| ADOT sidecar | in each task | OTLP -> X-Ray traces + CloudWatch |
| SSM SecureString | 4 params | DB URL, signing key, API keys |
| VPC with 2 public subnets | no NAT gateway | keeps the bill near zero |

After `apply`, run the one-shot jobs once (from CI or locally with the RDS URL): `ssa-seed`, then
`ssa-ingest data/policies --tenant store-a`. Enable the `vector` extension on RDS first
(`CREATE EXTENSION vector;` - schema.sql does this).

Kill switch on AWS: `terraform apply -var kill_switch=true` (redeploys with `KILL_SWITCH=true`), or
exec into the task and `touch /tmp/ssa.KILL` for an instant, no-redeploy stop.
