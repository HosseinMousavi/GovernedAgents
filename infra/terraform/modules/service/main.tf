variable "name" {}
variable "cluster_id" {}
variable "image" {}
variable "command" { type = list(string) }
variable "port" { type = number }
variable "env" { type = list(object({ name = string, value = string })) }
variable "secrets" { type = list(object({ name = string, valueFrom = string })) }
variable "exec_role_arn" {}
variable "task_role_arn" {}
variable "subnets" { type = list(string) }
variable "sg_id" {}
variable "log_group" {}
variable "region" {}
variable "sidecar" { type = any }
variable "registry_arn" { default = null }

resource "aws_ecs_task_definition" "task" {
  family                   = var.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = 256
  memory                   = 512
  execution_role_arn       = var.exec_role_arn
  task_role_arn            = var.task_role_arn
  container_definitions = jsonencode([
    {
      name         = "app"
      image        = var.image
      command      = var.command
      essential    = true
      portMappings = [{ containerPort = var.port, protocol = "tcp" }]
      environment  = var.env
      secrets      = var.secrets
      logConfiguration = { logDriver = "awslogs", options = { awslogs-group = var.log_group, awslogs-region = var.region, awslogs-stream-prefix = "app" } }
    },
    var.sidecar
  ])
}

resource "aws_ecs_service" "svc" {
  name            = var.name
  cluster         = var.cluster_id
  task_definition = aws_ecs_task_definition.task.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  network_configuration {
    subnets          = var.subnets
    security_groups  = [var.sg_id]
    assign_public_ip = true
  }
  dynamic "service_registries" {
    for_each = var.registry_arn == null ? [] : [var.registry_arn]
    content { registry_arn = service_registries.value }
  }
}
