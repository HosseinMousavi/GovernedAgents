# Store Support Agent on AWS - free-tier sized.
#   ECS Fargate (mcp + agents, 0.25 vCPU / 0.5 GB each) + RDS Postgres db.t3.micro (pgvector)
#   ADOT collector sidecar -> X-Ray traces + CloudWatch logs. Secrets in SSM Parameter Store.
#   No Ollama on AWS: embeddings fall back to OpenAI (FallbackEmbedder). n8n stays local / n8n cloud.
terraform {
  required_version = ">= 1.6"
  required_providers { aws = { source = "hashicorp/aws", version = "~> 5.60" } }
  backend "s3" {}   # configure with -backend-config (bucket, key, region) in CI
}

provider "aws" { region = var.region }

data "aws_availability_zones" "azs" {}

# ---------------- network (2 public subnets, no NAT = no NAT charges) ----------------
resource "aws_vpc" "main" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  tags                 = { Name = "${var.name}-vpc" }
}
resource "aws_internet_gateway" "igw" { vpc_id = aws_vpc.main.id }
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.azs.names[count.index]
  map_public_ip_on_launch = true
}
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route { cidr_block = "0.0.0.0/0", gateway_id = aws_internet_gateway.igw.id }
}
resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "svc" {
  name   = "${var.name}-svc"
  vpc_id = aws_vpc.main.id
  ingress { from_port = 8080, to_port = 8080, protocol = "tcp", cidr_blocks = [var.allowed_cidr] }
  ingress { from_port = 8765, to_port = 8765, protocol = "tcp", cidr_blocks = [var.allowed_cidr] }
  ingress { from_port = 0, to_port = 0, protocol = "-1", self = true }
  egress  { from_port = 0, to_port = 0, protocol = "-1", cidr_blocks = ["0.0.0.0/0"] }
}
resource "aws_security_group" "db" {
  name   = "${var.name}-db"
  vpc_id = aws_vpc.main.id
  ingress { from_port = 5432, to_port = 5432, protocol = "tcp", security_groups = [aws_security_group.svc.id] }
  egress  { from_port = 0, to_port = 0, protocol = "-1", cidr_blocks = ["0.0.0.0/0"] }
}

# ---------------- database ----------------
resource "aws_db_subnet_group" "db" {
  name       = "${var.name}-db"
  subnet_ids = aws_subnet.public[*].id
}
resource "aws_db_instance" "pg" {
  identifier             = "${var.name}-pg"
  engine                 = "postgres"
  engine_version         = "16"
  instance_class         = "db.t3.micro"        # free tier
  allocated_storage      = 20
  db_name                = "ssa"
  username               = "ssa"
  password               = var.db_password
  db_subnet_group_name   = aws_db_subnet_group.db.name
  vpc_security_group_ids = [aws_security_group.db.id]
  skip_final_snapshot    = true
  publicly_accessible    = false
  backup_retention_period = 1
}

# ---------------- registry + secrets ----------------
resource "aws_ecr_repository" "app" {
  name                 = var.name
  image_tag_mutability = "MUTABLE"
  force_delete         = true
}
locals {
  database_url = "postgresql://ssa:${var.db_password}@${aws_db_instance.pg.address}:5432/ssa"
  secrets = {
    DATABASE_URL      = local.database_url
    TOKEN_SIGNING_KEY = var.token_signing_key
    ANTHROPIC_API_KEY = var.anthropic_api_key
    OPENAI_API_KEY    = var.openai_api_key
  }
}
resource "aws_ssm_parameter" "secret" {
  for_each = local.secrets
  name     = "/${var.name}/${each.key}"
  type     = "SecureString"
  value    = each.value
}

# ---------------- ECS ----------------
resource "aws_ecs_cluster" "main" { name = var.name }
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.name}"
  retention_in_days = 7
}
data "aws_iam_policy_document" "assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals { type = "Service", identifiers = ["ecs-tasks.amazonaws.com"] }
  }
}
resource "aws_iam_role" "exec" {
  name               = "${var.name}-exec"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}
resource "aws_iam_role_policy_attachment" "exec" {
  role       = aws_iam_role.exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}
resource "aws_iam_role_policy" "exec_ssm" {
  role = aws_iam_role.exec.id
  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [{ Effect = "Allow", Action = ["ssm:GetParameters"], Resource = [for p in aws_ssm_parameter.secret : p.arn] }]
  })
}
resource "aws_iam_role" "task" {
  name               = "${var.name}-task"
  assume_role_policy = data.aws_iam_policy_document.assume.json
}
resource "aws_iam_role_policy_attachment" "task_otel" {
  role       = aws_iam_role.task.name
  policy_arn = "arn:aws:iam::aws:policy/AWSXrayWriteOnlyAccess"
}
resource "aws_iam_role_policy_attachment" "task_cw" {
  role       = aws_iam_role.task.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchAgentServerPolicy"
}

locals {
  image   = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
  secrets_list = [for k, p in aws_ssm_parameter.secret : { name = k, valueFrom = p.arn }]
  common_env = [
    { name = "KILL_SWITCH", value = tostring(var.kill_switch) },
    { name = "OTEL_EXPORTER_OTLP_ENDPOINT", value = "http://localhost:4317" },
    { name = "OTEL_SERVICE_NAME", value = var.name },
    { name = "OLLAMA_BASE_URL", value = "http://127.0.0.1:1" },  # unreachable on AWS -> OpenAI fallback
    { name = "MCP_SERVER_URL", value = "http://mcp.${var.name}.local:8765/mcp" },
  ]
  otel_sidecar = {
    name      = "aws-otel-collector"
    image     = "public.ecr.aws/aws-observability/aws-otel-collector:latest"
    essential = true
    command   = ["--config=/etc/ecs/ecs-xray.yaml"]   # OTLP in -> X-Ray traces + CloudWatch metrics
    logConfiguration = { logDriver = "awslogs", options = { awslogs-group = aws_cloudwatch_log_group.app.name, awslogs-region = var.region, awslogs-stream-prefix = "otel" } }
  }
}

resource "aws_service_discovery_private_dns_namespace" "ns" {
  name = "${var.name}.local"
  vpc  = aws_vpc.main.id
}
resource "aws_service_discovery_service" "mcp" {
  name = "mcp"
  dns_config {
    namespace_id = aws_service_discovery_private_dns_namespace.ns.id
    dns_records { ttl = 10, type = "A" }
  }
}

module "service" {
  source   = "./modules/service"
  for_each = {
    mcp    = { port = 8765, command = ["python", "-m", "store_support.mcp_server.server"], registry = aws_service_discovery_service.mcp.arn }
    agents = { port = 8080, command = ["python", "-m", "store_support.resolver.service"], registry = null }
  }
  name          = "${var.name}-${each.key}"
  cluster_id    = aws_ecs_cluster.main.id
  image         = local.image
  command       = each.value.command
  port          = each.value.port
  env           = concat(local.common_env, [{ name = "HEALTH_URL", value = "http://localhost:${each.value.port}/healthz" }])
  secrets       = local.secrets_list
  exec_role_arn = aws_iam_role.exec.arn
  task_role_arn = aws_iam_role.task.arn
  subnets       = aws_subnet.public[*].id
  sg_id         = aws_security_group.svc.id
  log_group     = aws_cloudwatch_log_group.app.name
  region        = var.region
  sidecar       = local.otel_sidecar
  registry_arn  = each.value.registry
}
