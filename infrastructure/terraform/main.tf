##################################################################
# QuantEdge v6.0 — Complete AWS Infrastructure
# quant.dileepkapu.com | Owner: Dileep Kumar Reddy Kapu
##################################################################
terraform {
  required_version = ">= 1.7.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }
  backend "s3" {
    bucket         = "quantedge-terraform-state-dileep"
    key            = "quantedge/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "quantedge-terraform-locks"
    encrypt        = true
  }
}

# Primary provider
provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project   = "QuantEdge"
      Owner     = "Dileep Kumar Reddy Kapu"
      ManagedBy = "Terraform"
      Env       = var.environment
    }
  }
}

# WAF + ACM for CloudFront MUST be us-east-1
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
  default_tags {
    tags = {
      Project   = "QuantEdge"
      Owner     = "Dileep Kumar Reddy Kapu"
      ManagedBy = "Terraform"
      Env       = var.environment
    }
  }
}

###────────────────────────────────────────────────────────────###
#  VARIABLES
###────────────────────────────────────────────────────────────###
variable "aws_region" { default = "us-east-1" }
variable "environment" { default = "production" }
variable "domain_name" { default = "dileepkapu.com" }
variable "subdomain" { default = "quant" }
variable "owner_email" { default = "dileep@dileepkapu.com" }
variable "db_password" { sensitive = true }
variable "secret_key" { sensitive = true }
variable "anthropic_api_key" { sensitive = true }

locals {
  fqdn        = "${var.subdomain}.${var.domain_name}"
  account_id  = data.aws_caller_identity.current.account_id
  name_prefix = "quantedge"
}

###────────────────────────────────────────────────────────────###
#  DATA SOURCES
###────────────────────────────────────────────────────────────###
data "aws_caller_identity" "current" {}
data "aws_availability_zones" "available" { state = "available" }
data "aws_route53_zone" "main" { name = var.domain_name }

###────────────────────────────────────────────────────────────###
#  VPC
###────────────────────────────────────────────────────────────###
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags                 = { Name = "${local.name_prefix}-vpc" }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${local.name_prefix}-public-${count.index + 1}" }
}

resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags              = { Name = "${local.name_prefix}-private-${count.index + 1}" }
}

resource "aws_internet_gateway" "igw" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name_prefix}-igw" }
}

resource "aws_eip" "nat" { domain = "vpc" }

resource "aws_nat_gateway" "nat" {
  allocation_id = aws_eip.nat.id
  subnet_id     = aws_subnet.public[0].id
  depends_on    = [aws_internet_gateway.igw]
  tags          = { Name = "${local.name_prefix}-nat" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.igw.id
  }
  tags = { Name = "${local.name_prefix}-public-rt" }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.nat.id
  }
  tags = { Name = "${local.name_prefix}-private-rt" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

###────────────────────────────────────────────────────────────###
#  SECURITY GROUPS
###────────────────────────────────────────────────────────────###
resource "aws_security_group" "alb" {
  name   = "${local.name_prefix}-alb-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "${local.name_prefix}-alb-sg" }
}

resource "aws_security_group" "ecs" {
  name   = "${local.name_prefix}-ecs-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 8000
    to_port         = 8000
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = { Name = "${local.name_prefix}-ecs-sg" }
}

resource "aws_security_group" "rds" {
  name   = "${local.name_prefix}-rds-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }
  tags = { Name = "${local.name_prefix}-rds-sg" }
}

resource "aws_security_group" "redis" {
  name   = "${local.name_prefix}-redis-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }
  tags = { Name = "${local.name_prefix}-redis-sg" }
}

###────────────────────────────────────────────────────────────###
#  ACM SSL CERTIFICATE (us-east-1 required for CloudFront)
###────────────────────────────────────────────────────────────###
resource "aws_acm_certificate" "quantedge" {
  provider          = aws.us_east_1
  domain_name       = local.fqdn
  validation_method = "DNS"
  lifecycle { create_before_destroy = true }
}

resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.quantedge.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }
  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = data.aws_route53_zone.main.zone_id
}

resource "aws_acm_certificate_validation" "quantedge" {
  provider                = aws.us_east_1
  certificate_arn         = aws_acm_certificate.quantedge.arn
  validation_record_fqdns = [for r in aws_route53_record.cert_validation : r.fqdn]
}

###────────────────────────────────────────────────────────────###
#  APPLICATION LOAD BALANCER
###────────────────────────────────────────────────────────────###
resource "aws_lb" "quantedge" {
  name                       = "${local.name_prefix}-alb"
  internal                   = false
  load_balancer_type         = "application"
  security_groups            = [aws_security_group.alb.id]
  subnets                    = aws_subnet.public[*].id
  enable_deletion_protection = true


  tags = { Name = "${local.name_prefix}-alb" }
}

resource "aws_lb_target_group" "api" {
  name        = "${local.name_prefix}-api-tg"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/health"
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 10
    interval            = 30
    matcher             = "200"
  }
  tags = { Name = "${local.name_prefix}-api-tg" }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.quantedge.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate_validation.quantedge.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.quantedge.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

###────────────────────────────────────────────────────────────###
#  COGNITO
###────────────────────────────────────────────────────────────###
resource "aws_cognito_user_pool" "quantedge" {
  name = "${local.name_prefix}-users"

  password_policy {
    minimum_length                   = 16
    require_lowercase                = true
    require_uppercase                = true
    require_numbers                  = true
    require_symbols                  = true
    temporary_password_validity_days = 1
  }

  mfa_configuration = "ON"

  software_token_mfa_configuration {
    enabled = true
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  auto_verified_attributes = ["email"]

  user_pool_add_ons {
    advanced_security_mode = "ENFORCED"
  }

  username_attributes = []

  username_configuration {
    case_sensitive = false
  }

  schema {
    attribute_data_type = "String"
    name                = "email"
    required            = true
    mutable             = true
    string_attribute_constraints {
      min_length = 5
      max_length = 100
    }
  }

  tags = { Name = "${local.name_prefix}-user-pool" }
}

resource "aws_cognito_user_pool_client" "quantedge" {
  name         = "${local.name_prefix}-web-client"
  user_pool_id = aws_cognito_user_pool.quantedge.id

  explicit_auth_flows = [
    "ALLOW_USER_PASSWORD_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
    "ALLOW_USER_SRP_AUTH",
  ]

  access_token_validity  = 8
  id_token_validity      = 8
  refresh_token_validity = 30

  token_validity_units {
    access_token  = "hours"
    id_token      = "hours"
    refresh_token = "days"
  }

  allowed_oauth_flows_user_pool_client = false
  generate_secret                      = false
  prevent_user_existence_errors        = "ENABLED"
}

###────────────────────────────────────────────────────────────###
#  RDS POSTGRESQL
###────────────────────────────────────────────────────────────###
resource "aws_db_subnet_group" "quantedge" {
  name       = "${local.name_prefix}-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id
  tags       = { Name = "${local.name_prefix}-db-subnet-group" }
}

resource "aws_db_instance" "quantedge" {
  identifier                      = "${local.name_prefix}-postgres"
  engine                          = "postgres"
  engine_version                  = "15"
  instance_class                  = "db.t3.small"
  allocated_storage               = 50
  max_allocated_storage           = 200
  storage_encrypted               = true
  db_name                         = "quantedge"
  username                        = "quantedge_admin"
  password                        = var.db_password
  db_subnet_group_name            = aws_db_subnet_group.quantedge.name
  vpc_security_group_ids          = [aws_security_group.rds.id]
  multi_az                        = false
  publicly_accessible             = false
  backup_retention_period         = 7
  backup_window                   = "03:00-04:00"
  maintenance_window              = "Mon:04:00-Mon:05:00"
  deletion_protection             = true
  skip_final_snapshot             = false
  final_snapshot_identifier       = "${local.name_prefix}-final-snapshot"
  performance_insights_enabled    = true
  enabled_cloudwatch_logs_exports = ["postgresql", "upgrade"]
  tags                            = { Name = "${local.name_prefix}-postgres" }
}

###────────────────────────────────────────────────────────────###
#  ELASTICACHE REDIS
###────────────────────────────────────────────────────────────###
resource "aws_elasticache_subnet_group" "quantedge" {
  name       = "${local.name_prefix}-redis-subnet"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id = "${local.name_prefix}-redis"
  description          = "QuantEdge Redis"
  engine               = "redis"
  engine_version       = "7.1"
  node_type            = "cache.t3.micro"
  num_cache_clusters   = 1
  port                 = 6379
  parameter_group_name = "default.redis7"
  subnet_group_name    = aws_elasticache_subnet_group.quantedge.name
  security_group_ids   = [aws_security_group.redis.id]

  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  automatic_failover_enabled = false

  tags = { Name = "${local.name_prefix}-redis" }
}

###────────────────────────────────────────────────────────────###
#  ECR
###────────────────────────────────────────────────────────────###
resource "aws_ecr_repository" "api" {
  name                 = "${local.name_prefix}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
  tags = { Name = "${local.name_prefix}-api" }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection    = { tagStatus = "any", countType = "imageCountMoreThan", countNumber = 5 }
      action       = { type = "expire" }
    }]
  })
}

###────────────────────────────────────────────────────────────###
#  IAM
###────────────────────────────────────────────────────────────###
resource "aws_iam_role" "ecs_task" {
  name = "${local.name_prefix}-ecs-task-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "${local.name_prefix}-ecs-task-policy"
  role = aws_iam_role.ecs_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "CognitoAccess"
        Effect   = "Allow"
        Action   = ["cognito-idp:GetUser", "cognito-idp:InitiateAuth", "cognito-idp:RespondToAuthChallenge", "cognito-idp:AdminCreateUser", "cognito-idp:AdminSetUserMFAPreference"]
        Resource = aws_cognito_user_pool.quantedge.arn
      },
      {
        Sid      = "S3Access"
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = ["${aws_s3_bucket.datalake.arn}/*", "${aws_s3_bucket.datalake.arn}", "${aws_s3_bucket.models.arn}/*", "${aws_s3_bucket.models.arn}"]
      },
      { Sid = "SecretsAccess", Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = "arn:aws:secretsmanager:${var.aws_region}:${local.account_id}:secret:quantedge/*" },
      { Sid = "SNSAlerts", Effect = "Allow", Action = ["sns:Publish"], Resource = aws_sns_topic.alerts.arn },
      { Sid = "SageMaker", Effect = "Allow", Action = ["sagemaker:InvokeEndpoint"], Resource = "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:endpoint/quantedge-*" },
      { Sid = "CloudWatchLogs", Effect = "Allow", Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"], Resource = "*" }
    ]
  })
}

resource "aws_iam_role" "ecs_execution" {
  name = "${local.name_prefix}-ecs-execution-role"
  assume_role_policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Action = "sts:AssumeRole", Effect = "Allow", Principal = { Service = "ecs-tasks.amazonaws.com" } }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_secrets" {
  name = "${local.name_prefix}-ecs-execution-secrets"
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version   = "2012-10-17"
    Statement = [{ Effect = "Allow", Action = ["secretsmanager:GetSecretValue"], Resource = "arn:aws:secretsmanager:${var.aws_region}:${local.account_id}:secret:quantedge/*" }]
  })
}

###────────────────────────────────────────────────────────────###
#  ECS CLUSTER + FARGATE
###────────────────────────────────────────────────────────────###
resource "aws_ecs_cluster" "quantedge" {
  name = "${local.name_prefix}-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  tags = { Name = "${local.name_prefix}-cluster" }
}

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${local.name_prefix}"
  retention_in_days = 30
}

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name_prefix}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([{
    name         = "quantedge-api"
    image        = "${aws_ecr_repository.api.repository_url}:latest"
    essential    = true
    portMappings = [{ containerPort = 8000, protocol = "tcp" }]
    environment = [
      { name = "APP_ENV", value = "production" },
      { name = "AWS_REGION", value = var.aws_region },
      { name = "AWS_ACCOUNT_ID", value = local.account_id },
      { name = "COGNITO_USER_POOL_ID", value = aws_cognito_user_pool.quantedge.id },
      { name = "COGNITO_CLIENT_ID", value = aws_cognito_user_pool_client.quantedge.id },
      { name = "REDIS_URL", value = "rediss://${aws_elasticache_replication_group.redis.primary_endpoint_address}:6379" },
      { name = "CORS_ORIGINS", value = "https://${local.fqdn}" },
      { name = "DATABASE_URL", value = "postgresql+asyncpg://quantedge_admin:${var.db_password}@${aws_db_instance.quantedge.endpoint}/quantedge" },
      { name = "S3_BUCKET_DATA", value = aws_s3_bucket.datalake.bucket },
      { name = "S3_BUCKET_MODELS", value = aws_s3_bucket.models.bucket },
      { name = "SNS_ALERT_TOPIC_ARN", value = aws_sns_topic.alerts.arn },
      { name = "OWNER_USERNAME", value = "dileep" },
      { name = "USE_ANTHROPIC_FALLBACK", value = "true" },
    ]
    secrets = [
      { name = "SECRET_KEY", valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:SECRET_KEY::" },
      { name = "ALPHA_VANTAGE_KEY", valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:ALPHA_VANTAGE_KEY::" },
      { name = "FRED_API_KEY", valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:FRED_API_KEY::" },
      { name = "ANTHROPIC_API_KEY", valueFrom = "${aws_secretsmanager_secret.app_secrets.arn}:ANTHROPIC_API_KEY::" },
    ]
    logConfiguration = {
      logDriver = "awslogs"
      options   = { "awslogs-group" = aws_cloudwatch_log_group.ecs.name, "awslogs-region" = var.aws_region, "awslogs-stream-prefix" = "api" }
    }
    healthCheck = { command = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"], interval = 30, timeout = 10, retries = 3, startPeriod = 60 }
  }])
}

resource "aws_ecs_service" "api" {
  name            = "${local.name_prefix}-api"
  cluster         = aws_ecs_cluster.quantedge.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = 1
  launch_type     = "FARGATE"
  health_check_grace_period_seconds = 180

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "quantedge-api"
    container_port   = 8000
  }

  lifecycle { ignore_changes = [desired_count, task_definition] }
  depends_on = [aws_lb_listener.https]
}

###────────────────────────────────────────────────────────────###
#  S3 BUCKETS
###────────────────────────────────────────────────────────────###
resource "aws_s3_bucket" "frontend" {
  bucket = "${local.name_prefix}-frontend-dileep"
  tags   = { Name = "${local.name_prefix}-frontend" }
}

resource "aws_s3_bucket_website_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id
  index_document { suffix = "index.html" }
  error_document { key = "index.html" }
}

resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket                  = aws_s3_bucket.frontend.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "frontend" {
  bucket = aws_s3_bucket.frontend.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "AllowCloudFrontOAC"
      Effect    = "Allow"
      Principal = { Service = "cloudfront.amazonaws.com" }
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.frontend.arn}/*"
      Condition = { StringEquals = { "AWS:SourceArn" = aws_cloudfront_distribution.quantedge.arn } }
    }]
  })
}

resource "aws_s3_bucket" "datalake" {
  bucket = "${local.name_prefix}-datalake-dileep"
  tags   = { Name = "${local.name_prefix}-datalake" }
}

resource "aws_s3_bucket_versioning" "datalake" {
  bucket = aws_s3_bucket.datalake.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket" "models" {
  bucket = "${local.name_prefix}-models-dileep"
  tags   = { Name = "${local.name_prefix}-ml-models" }
}

resource "aws_s3_bucket" "alb_logs" {
  bucket = "${local.name_prefix}-alb-logs-dileep"
  tags   = { Name = "${local.name_prefix}-alb-logs" }
}

resource "aws_s3_bucket_public_access_block" "datalake" {
  bucket                  = aws_s3_bucket.datalake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_public_access_block" "models" {
  bucket                  = aws_s3_bucket.models.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

###────────────────────────────────────────────────────────────###
#  CLOUDFRONT — Origin Request Policy (replaces forwarding_config)
###────────────────────────────────────────────────────────────###
resource "aws_cloudfront_origin_request_policy" "api_passthrough" {
  provider = aws.us_east_1
  name     = "${local.name_prefix}-api-passthrough"
  cookies_config { cookie_behavior = "all" }
  headers_config {
    header_behavior = "whitelist"
    headers { items = ["Content-Type", "Origin", "Accept"] }
  }
  query_strings_config { query_string_behavior = "all" }
}

###────────────────────────────────────────────────────────────###
#  CLOUDFRONT CDN
###────────────────────────────────────────────────────────────###
resource "aws_cloudfront_origin_access_control" "frontend" {
  provider                          = aws.us_east_1
  name                              = "${local.name_prefix}-oac"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_distribution" "quantedge" {
  provider            = aws.us_east_1
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  aliases             = [local.fqdn]
  price_class         = "PriceClass_100"
  # web_acl_id          = aws_wafv2_web_acl.quantedge.arn

  origin {
    domain_name              = aws_s3_bucket.frontend.bucket_regional_domain_name
    origin_id                = "S3Frontend"
    origin_access_control_id = aws_cloudfront_origin_access_control.frontend.id
  }

  origin {
    domain_name = aws_lb.quantedge.dns_name
    origin_id   = "ALBBackend"
    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "https-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  # /api/* → ALB, no caching, forward auth headers
  ordered_cache_behavior {
    path_pattern             = "/api/*"
    allowed_methods          = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods           = ["GET", "HEAD"]
    target_origin_id         = "ALBBackend"
    cache_policy_id          = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad" # CachingDisabled
    origin_request_policy_id = aws_cloudfront_origin_request_policy.api_passthrough.id
    viewer_protocol_policy   = "https-only"
    compress                 = true
  }

  # /auth/* → ALB
  ordered_cache_behavior {
    path_pattern             = "/auth/*"
    allowed_methods          = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods           = ["GET", "HEAD"]
    target_origin_id         = "ALBBackend"
    cache_policy_id          = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
    origin_request_policy_id = aws_cloudfront_origin_request_policy.api_passthrough.id
    viewer_protocol_policy   = "https-only"
    compress                 = true
  }

  # /ws/* → ALB (WebSocket)
  ordered_cache_behavior {
    path_pattern             = "/ws/*"
    allowed_methods          = ["GET", "HEAD", "OPTIONS"]
    cached_methods           = ["GET", "HEAD"]
    target_origin_id         = "ALBBackend"
    cache_policy_id          = "4135ea2d-6df8-44a3-9df3-4b5a84be39ad"
    origin_request_policy_id = aws_cloudfront_origin_request_policy.api_passthrough.id
    viewer_protocol_policy   = "https-only"
    compress                 = false
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3Frontend"
    viewer_protocol_policy = "redirect-to-https"
    compress               = true
    cache_policy_id        = "658327ea-f89d-4fab-a63d-7e88639e58f6" # CachingOptimized
  }

  custom_error_response {
    error_code         = 404
    response_code      = 200
    response_page_path = "/index.html"
  }
  custom_error_response {
    error_code         = 403
    response_code      = 200
    response_page_path = "/index.html"
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.quantedge.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  restrictions {
    geo_restriction { restriction_type = "none" }
  }

  tags = { Name = "${local.name_prefix}-cdn" }
}

###────────────────────────────────────────────────────────────###
#  ROUTE 53
###────────────────────────────────────────────────────────────###
resource "aws_route53_record" "quantedge" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = local.fqdn
  type    = "A"
  alias {
    name                   = aws_cloudfront_distribution.quantedge.domain_name
    zone_id                = aws_cloudfront_distribution.quantedge.hosted_zone_id
    evaluate_target_health = false
  }
}

###────────────────────────────────────────────────────────────###
#  SECRETS MANAGER
###────────────────────────────────────────────────────────────###
resource "aws_secretsmanager_secret" "app_secrets" {
  name                    = "quantedge/app-secrets"
  recovery_window_in_days = 7
  tags                    = { Name = "${local.name_prefix}-secrets" }
}

resource "aws_secretsmanager_secret_version" "app_secrets" {
  secret_id = aws_secretsmanager_secret.app_secrets.id
  secret_string = jsonencode({
    SECRET_KEY           = var.secret_key
    ALPHA_VANTAGE_KEY    = "YOUR_ALPHA_VANTAGE_KEY_HERE"
    FRED_API_KEY         = "YOUR_FRED_API_KEY_HERE"
    ANTHROPIC_API_KEY    = var.anthropic_api_key
    REDDIT_CLIENT_ID     = "YOUR_REDDIT_CLIENT_ID_HERE"
    REDDIT_CLIENT_SECRET = "YOUR_REDDIT_CLIENT_SECRET_HERE"
  })
  lifecycle { ignore_changes = [secret_string] }
}

###────────────────────────────────────────────────────────────###
#  SNS + CLOUDWATCH ALARMS
###────────────────────────────────────────────────────────────###
resource "aws_sns_topic" "alerts" {
  name = "${local.name_prefix}-alerts"
  tags = { Name = "${local.name_prefix}-alerts" }
}

resource "aws_sns_topic_subscription" "dileep_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.owner_email
}

resource "aws_cloudwatch_metric_alarm" "ecs_cpu_high" {
  alarm_name          = "${local.name_prefix}-ecs-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/ECS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_actions       = [aws_sns_topic.alerts.arn]
  dimensions          = { ClusterName = aws_ecs_cluster.quantedge.name, ServiceName = aws_ecs_service.api.name }
}

resource "aws_cloudwatch_metric_alarm" "rds_cpu_high" {
  alarm_name          = "${local.name_prefix}-rds-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/RDS"
  period              = 300
  statistic           = "Average"
  threshold           = 80
  alarm_actions       = [aws_sns_topic.alerts.arn]
  dimensions          = { DBInstanceIdentifier = aws_db_instance.quantedge.id }
}

###────────────────────────────────────────────────────────────###
#  OUTPUTS
###────────────────────────────────────────────────────────────###
output "cloudfront_url" { value = "https://${local.fqdn}" }
output "cloudfront_domain" { value = aws_cloudfront_distribution.quantedge.domain_name }
output "cloudfront_distribution_id" { value = aws_cloudfront_distribution.quantedge.id }
output "alb_dns" { value = aws_lb.quantedge.dns_name }
output "ecr_repository_url" { value = aws_ecr_repository.api.repository_url }
output "cognito_user_pool_id" { value = aws_cognito_user_pool.quantedge.id }
output "cognito_client_id" { value = aws_cognito_user_pool_client.quantedge.id }
output "rds_endpoint" { value = aws_db_instance.quantedge.endpoint }
output "redis_endpoint" { value = aws_elasticache_replication_group.redis.primary_endpoint_address }
output "s3_frontend_bucket" { value = aws_s3_bucket.frontend.bucket }
output "s3_datalake_bucket" { value = aws_s3_bucket.datalake.bucket }
output "sns_topic_arn" { value = aws_sns_topic.alerts.arn }
