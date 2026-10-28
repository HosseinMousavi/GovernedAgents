output "ecr_repository" { value = aws_ecr_repository.app.repository_url }
output "db_endpoint"    { value = aws_db_instance.pg.address }
output "cluster"        { value = aws_ecs_cluster.main.name }
output "log_group"      { value = aws_cloudwatch_log_group.app.name }
