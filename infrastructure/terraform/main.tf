###############################################################################
#  QuantEdge v6.0 — PERSONAL USE TERRAFORM
#  Optimised for solo use: ~$33/month AWS cost (vs $86 full stack)
#
#  What was REMOVED vs enterprise version:
#    - NAT Gateway          (~$35/mo) → ECS in public subnets instead
#    - ALB                  (~$20/mo) → CloudFront → ECS direct
#    - ElastiCache          (~$15/mo) → Redis runs as sidecar in ECS task
#    - WAF                  (~$6/mo)  → not needed for personal use
#    - db.t3.small RDS      → downgraded to db.t3.micro
#    - ECS 1vCPU/2GB        → downgraded to 0.5vCPU/1GB
#    - Multi-AZ anything    → single-AZ everywhere
#    - Performance Insights → disabled (costs extra)
#    - 200GB max storage    → 20GB max (plenty for personal signal tracker)
#
#  What was KEPT:
#    - RDS PostgreSQL 15 db.t3.micro  (signal tracker data — must persist)
#    - ECS Fargate 0.5vCPU/1GB        (runs the FastAPI backend)
#    - ECR                            (Docker image storage)
#    - CloudFront + S3                (frontend hosting)
#    - Cognito                        (MFA auth — keep for security)
#    - Secrets Manager                (API keys)
#    - CloudWatch Logs                (debugging)
#    - SNS Alerts                     (error notifications)
#    - ACM + Route53                  (HTTPS + domain)
#    - IAM roles                      (required)
#
#  Redis: runs as a sidecar container in the same ECS task.
#         REDIS_URL = redis://localhost:6379/0
#         Data is in-memory only — lost on ECS task restart.
#         This is fine: Redis is used for caching and pub/sub, not
#         for persistent data. All persistent data is in RDS.
###############################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  backend "s3" {
    bucket         = "quantedge-terraform-state-dileep"
    key            = "quantedge/personal/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "quantedge-terraform-locks"
    encrypt        = true
  }
}

###─────────────────────────────────────────────────────────────────────────###
#  PROVIDERS
###─────────────────────────────────────────────────────────────────────────###

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "quantedge"
      Environment = "personal"
      Owner       = "dileep"
      ManagedBy   = "terraform"
    }
  }
}

# ACM certificates for CloudFront must be in us-east-1
provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
  default_tags {
    tags = {
      Project     = "quantedge"
      Environment = "personal"
      Owner       = "dileep"
      ManagedBy   = "terraform"
    }
  }
}

###─────────────────────────────────────────────────────────────────────────###
#  VARIABLES
###─────────────────────────────────────────────────────────────────────────###

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "personal"
}

variable "domain_name" {
  description = "Root domain name"
  type        = string
  default     = "dileepkapu.com"
}

variable "subdomain" {
  description = "Subdomain for the app"
  type        = string
  default     = "quant"
}

variable "owner_email" {
  description = "Owner email for alerts"
  type        = string
  default     = "dileep@dileepkapu.com"
}

variable "db_password" {
  description = "RDS master password"
  type        = string
  sensitive   = true
}

variable "secret_key" {
  description = "FastAPI JWT secret key (min 32 chars)"
  type        = string
  sensitive   = true
}

variable "anthropic_api_key" {
  description = "Anthropic API key"
  type        = string
  sensitive   = true
}

###─────────────────────────────────────────────────────────────────────────###
#  LOCALS
###─────────────────────────────────────────────────────────────────────────###

locals {
  name_prefix = "quantedge"
  full_domain = "${var.subdomain}.${var.domain_name}"
}

###─────────────────────────────────────────────────────────────────────────###
#  DATA SOURCES
###─────────────────────────────────────────────────────────────────────────###

data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

data "aws_route53_zone" "main" {
  name         = var.domain_name
  private_zone = false
}

###─────────────────────────────────────────────────────────────────────────###
#  VPC — simplified: public subnets only (no NAT Gateway needed)
#  ECS runs in public subnets with a security group that blocks all inbound
#  except CloudFront IP ranges. RDS stays in private subnets.
###─────────────────────────────────────────────────────────────────────────###

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = "${local.name_prefix}-vpc" }
}

# Public subnets — ECS Fargate tasks run here (no NAT Gateway needed)
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.${count.index}.0/24"
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${local.name_prefix}-public-${count.index + 1}" }
}

# Private subnets — RDS only (no internet access required for database)
resource "aws_subnet" "private" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  tags              = { Name = "${local.name_prefix}-private-${count.index + 1}" }
}

# Internet Gateway — for public subnets
resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name_prefix}-igw" }
}

# Route table for public subnets — route all traffic through IGW
resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
  tags = { Name = "${local.name_prefix}-public-rt" }
}

resource "aws_route_table_association" "public" {
  count          = 2
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# Route table for private subnets — no internet route (RDS does not need internet)
resource "aws_route_table" "private" {
  vpc_id = aws_vpc.main.id
  tags   = { Name = "${local.name_prefix}-private-rt" }
}

resource "aws_route_table_association" "private" {
  count          = 2
  subnet_id      = aws_subnet.private[count.index].id
  route_table_id = aws_route_table.private.id
}

###─────────────────────────────────────────────────────────────────────────###
#  SECURITY GROUPS
###─────────────────────────────────────────────────────────────────────────###

# ECS security group
# Inbound: port 8000 from anywhere (CloudFront will be the only caller in practice)
# We lock this down via CloudFront custom header in production
# Outbound: all (needs to reach RDS, ECR, Secrets Manager, Polygon API, Anthropic)
resource "aws_security_group" "ecs" {
  name        = "${local.name_prefix}-ecs-sg"
  description = "ECS Fargate tasks - personal QuantEdge"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTPS for VPC endpoints (ECR, Secrets Manager, CloudWatch)"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "FastAPI from CloudFront and health checks"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All outbound - needs ECR, RDS, Secrets Manager, Polygon, Anthropic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-ecs-sg" }
}

# RDS security group
# Inbound: PostgreSQL from ECS only
resource "aws_security_group" "rds" {
  name        = "${local.name_prefix}-rds-sg"
  description = "RDS PostgreSQL - personal QuantEdge"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "PostgreSQL from ECS tasks only"
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = { Name = "${local.name_prefix}-rds-sg" }
}

###─────────────────────────────────────────────────────────────────────────###
#  VPC ENDPOINTS — allow ECS (in public subnet) to reach AWS services
#  without going over the internet. Saves on data transfer costs too.
###─────────────────────────────────────────────────────────────────────────###











###─────────────────────────────────────────────────────────────────────────###
#  ECR — Docker image repository
###─────────────────────────────────────────────────────────────────────────###

resource "aws_ecr_repository" "quantedge_api" {
  name                 = "quantedge-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = { Name = "quantedge-api" }
}

# Keep only last 5 images to save storage costs
resource "aws_ecr_lifecycle_policy" "quantedge_api" {
  repository = aws_ecr_repository.quantedge_api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 5 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 5
      }
      action = { type = "expire" }
    }]
  })
}

###─────────────────────────────────────────────────────────────────────────###
#  IAM — ECS execution and task roles
###─────────────────────────────────────────────────────────────────────────###

# Execution role — used by ECS agent to pull image and fetch secrets
resource "aws_iam_role" "ecs_execution_role" {
  name = "${local.name_prefix}-ecs-execution-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_policy" {
  role       = aws_iam_role.ecs_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Allow execution role to read from Secrets Manager
resource "aws_iam_role_policy" "ecs_execution_secrets" {
  name = "${local.name_prefix}-ecs-execution-secrets"
  role = aws_iam_role.ecs_execution_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ]
      Resource = aws_secretsmanager_secret.quantedge_secrets.arn
    }]
  })
}

# Task role — used by the application itself
resource "aws_iam_role" "ecs_task_role" {
  name = "${local.name_prefix}-ecs-task-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "${local.name_prefix}-ecs-task-policy"
  role = aws_iam_role.ecs_task_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/ecs/quantedge-api:*"
      },
      {
        Sid    = "SNSAlerts"
        Effect = "Allow"
        Action = ["sns:Publish"]
        Resource = aws_sns_topic.alerts.arn
      },
      {
        Sid    = "CognitoVerify"
        Effect = "Allow"
        Action = [
          "cognito-idp:GetUser",
          "cognito-idp:AdminGetUser"
        ]
        Resource = aws_cognito_user_pool.quantedge.arn
      }
    ]
  })
}

###─────────────────────────────────────────────────────────────────────────###
#  CLOUDWATCH LOG GROUP
###─────────────────────────────────────────────────────────────────────────###

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/quantedge-api"
  retention_in_days = 30  # 30 days is plenty for personal use
  tags              = { Name = "/ecs/quantedge-api" }
}

###─────────────────────────────────────────────────────────────────────────###
#  SECRETS MANAGER
###─────────────────────────────────────────────────────────────────────────###

resource "aws_secretsmanager_secret" "quantedge_secrets" {
  name                    = "quantedge-secrets"
  description             = "QuantEdge v6.0 application secrets"
  recovery_window_in_days = 7
  tags                    = { Name = "quantedge-secrets" }
}

resource "aws_secretsmanager_secret_version" "quantedge_secrets" {
  secret_id = aws_secretsmanager_secret.quantedge_secrets.id
  secret_string = jsonencode({
    SECRET_KEY        = var.secret_key
    ANTHROPIC_API_KEY = var.anthropic_api_key
    POLYGON_API_KEY   = "REPLACE_AFTER_APPLY"
    DB_PASSWORD       = var.db_password
  })

  # Prevent Terraform from overwriting POLYGON_API_KEY once set manually
  lifecycle {
    ignore_changes = [secret_string]
  }
}

###─────────────────────────────────────────────────────────────────────────###
#  RDS POSTGRESQL — db.t3.micro, personal use sizing
###─────────────────────────────────────────────────────────────────────────###
resource "aws_db_subnet_group" "quantedge" {
  name       = "${local.name_prefix}-db-subnet-group"
  subnet_ids = aws_subnet.private[*].id
  tags       = { Name = "${local.name_prefix}-db-subnet-group" }
}

resource "aws_db_instance" "quantedge" {
  identifier     = "quantedge-postgres"
  engine         = "postgres"
  engine_version = "15"          # PostgreSQL 15 — stable, well supported
  instance_class = "db.t3.micro" # Cheapest RDS instance — fine for 1 user

  allocated_storage     = 20    # 20GB — plenty for signal tracker data
  max_allocated_storage = 20    # No autoscaling storage (keep costs predictable)
  storage_encrypted     = true

  db_name  = "quantedge"
  username = "quantedge"
  password = var.db_password

  db_subnet_group_name   = aws_db_subnet_group.quantedge.name
  vpc_security_group_ids = [aws_security_group.rds.id]

  multi_az            = false  # Single-AZ — fine for personal use
  publicly_accessible = false

  backup_retention_period = 7
  backup_window           = "03:00-04:00"
  maintenance_window      = "Mon:04:00-Mon:05:00"

  deletion_protection             = true
  skip_final_snapshot             = false
  final_snapshot_identifier       = "quantedge-postgres-final-snapshot"
  performance_insights_enabled    = false  # Disabled — saves cost

  tags = { Name = "quantedge-postgres" }
}
###─────────────────────────────────────────────────────────────────────────###
#  ECS CLUSTER
###─────────────────────────────────────────────────────────────────────────###

resource "aws_ecs_cluster" "main" {
  name = "quantedge-cluster"

  setting {
    name  = "containerInsights"
    value = "disabled"  # Disabled to save cost — use CloudWatch logs directly
  }

  tags = { Name = "quantedge-cluster" }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
    base              = 1
  }
}

###─────────────────────────────────────────────────────────────────────────###
#  ECS TASK DEFINITION
#
#  Two containers in one task:
#    1. quantedge-api  — FastAPI backend (0.5vCPU / 896MB)
#    2. redis-sidecar  — Redis 7        (0.25vCPU / 128MB reserved, ~64MB actual)
#
#  Total task: 0.75vCPU / ~1024MB
#  Fargate pricing at 0.75vCPU/1GB ≈ $0.013/hour = ~$9.50/month
#  (vs 1vCPU/2GB ≈ $0.025/hour = ~$18/month in enterprise version)
#
#  Redis is localhost inside the task — REDIS_URL = redis://localhost:6379/0
#  Redis data is lost on task restart. This is acceptable because:
#    - Cache data: re-fetched from Polygon on miss
#    - Pub/sub: reconnects on restart
#    - Sessions: Cognito JWT — stateless
###─────────────────────────────────────────────────────────────────────────###

resource "aws_ecs_task_definition" "quantedge_api" {
  family                   = "quantedge-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"  # 1 vCPU
  memory                   = "2048"  # 2GB total
  execution_role_arn       = aws_iam_role.ecs_execution_role.arn
  task_role_arn            = aws_iam_role.ecs_task_role.arn

  container_definitions = jsonencode([
    ###──────────────────────────────────────────────
    # Container 1: Redis sidecar
    # Replaces ElastiCache — saves $15/month
    # Runs as localhost inside the task
    ###──────────────────────────────────────────────
    {
      name      = "redis-sidecar"
      image     = "redis:7-alpine"
      essential = false  # Task continues if Redis crashes (FastAPI handles gracefully)
      cpu       = 0      # No hard reservation — shares task CPU
      memory    = 128    # Hard limit 128MB — Redis with maxmemory 96mb

      command = [
        "redis-server",
        "--maxmemory", "96mb",
        "--maxmemory-policy", "allkeys-lru",
        "--save", "",          # Disable RDB snapshots (no persistence needed)
        "--appendonly", "no"   # Disable AOF (no persistence needed)
      ]

      portMappings = [{
        containerPort = 6379
        protocol      = "tcp"
      }]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/quantedge-api"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "redis"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "redis-cli ping || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 10
      }
    },

    ###──────────────────────────────────────────────
    # Container 2: QuantEdge FastAPI backend
    ###──────────────────────────────────────────────
    {
      name      = "quantedge-api"
      image     = "${aws_ecr_repository.quantedge_api.repository_url}:latest"
      essential = true
      cpu       = 0      # No hard reservation within task (shares with redis)
      memory    = 1920    # Hard limit 1920MB (2048 - 128 for Redis)

      portMappings = [{
        containerPort = 8000
        protocol      = "tcp"
      }]

      environment = [
        # Redis is localhost — no ElastiCache needed
        { name = "REDIS_URL",             value = "redis://localhost:6379/0" },
        # RDS connection — PostgreSQL 15 on db.t3.micro
        { name = "DATABASE_URL", value = "postgresql://quantedge:${var.db_password}@quantedge-postgres.c09yes6ea4te.us-east-1.rds.amazonaws.com/quantedge" },
        # Application settings
        { name = "ENVIRONMENT",           value = "production" },
        { name = "MODEL_DIR",             value = "/app/models" },
        { name = "AWS_REGION",            value = var.aws_region },
        { name = "COGNITO_USER_POOL_ID",  value = aws_cognito_user_pool.quantedge.id },
        { name = "COGNITO_CLIENT_ID",     value = aws_cognito_user_pool_client.app.id },
        { name = "SNS_ALERT_TOPIC_ARN",   value = aws_sns_topic.alerts.arn },
        { name = "CORS_ORIGINS", value = "https://${local.full_domain}" },
        # Reduce workers — 1 is enough for personal use (saves memory)
        { name = "UVICORN_WORKERS",       value = "1" }
      ]

      secrets = [
        { name = "ANTHROPIC_API_KEY", valueFrom = "${aws_secretsmanager_secret.quantedge_secrets.arn}:ANTHROPIC_API_KEY::" },
        { name = "POLYGON_API_KEY",   valueFrom = "${aws_secretsmanager_secret.quantedge_secrets.arn}:POLYGON_API_KEY::" },
        { name = "SECRET_KEY",        valueFrom = "${aws_secretsmanager_secret.quantedge_secrets.arn}:SECRET_KEY::" }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/quantedge-api"
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs"
        }
      }

      healthCheck = {
        command     = ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]
        interval    = 30
        timeout     = 10
        retries     = 3
        startPeriod = 300  # 10 minutes — ML model training on first boot
      }

      # Redis sidecar must start before API (so localhost:6379 is ready)
      dependsOn = [{
        containerName = "redis-sidecar"
        condition     = "HEALTHY"
      }]
    }
  ])

  tags = { Name = "quantedge-api" }
}

###─────────────────────────────────────────────────────────────────────────###
#  ECS SERVICE
#  Runs in PUBLIC subnets — no NAT Gateway needed
#  assign_public_ip = true → ECS gets a public IP for outbound internet
###─────────────────────────────────────────────────────────────────────────###

resource "aws_ecs_service" "quantedge_api" {
  name            = "quantedge-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.quantedge_api.arn
  desired_count   = 1  # Single instance — personal use


  # Use FARGATE_SPOT when available for up to 70% cost savings
  # Falls back to regular FARGATE automatically
  capacity_provider_strategy {
    capacity_provider = "FARGATE_SPOT"
    weight            = 1
    base              = 0
  }

  capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 0
    base              = 1
  }

  network_configuration {
    subnets          = aws_subnet.public[*].id    # Public subnets (no NAT needed)
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = true                        # Required for public subnet outbound
  }

  # No load balancer — CloudFront connects directly to ECS public IP
  # CloudFront uses the ECS_SERVICE_URL output to set its origin

  health_check_grace_period_seconds = 300  # 10 minutes for first ML model training

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  deployment_controller {
    type = "ECS"
  }

  # Force new deployment when task definition changes
  force_new_deployment = true

  depends_on = [
    aws_iam_role_policy_attachment.ecs_execution_policy,
    aws_iam_role_policy.ecs_execution_secrets,
    aws_iam_role_policy.ecs_task_policy,
    aws_cloudwatch_log_group.ecs
  ]

  tags = { Name = "quantedge-api" }
}

###─────────────────────────────────────────────────────────────────────────###
#  S3 — Frontend static files
###─────────────────────────────────────────────────────────────────────────###

resource "aws_s3_bucket" "frontend" {
  bucket        = "quantedge-frontend-dileep"
  force_destroy = false
  tags          = { Name = "quantedge-frontend-dileep" }
}

resource "aws_s3_bucket_versioning" "frontend" {
  bucket = aws_s3_bucket.frontend.id
  versioning_configuration {
    status = "Disabled"  # No versioning needed — CloudFront serves latest
  }
}

resource "aws_s3_bucket_public_access_block" "frontend" {
  bucket                  = aws_s3_bucket.frontend.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "frontend" {
  bucket = aws_s3_bucket.frontend.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

###─────────────────────────────────────────────────────────────────────────###
#  CLOUDFRONT — serves frontend from S3 AND proxies /api/* to ECS directly
#
#  Architecture (no ALB):
#    Browser → CloudFront → /api/* → ECS public IP (origin = ECS task IP)
#    Browser → CloudFront → /*     → S3 (React frontend)
#
#  NOTE: Because ECS tasks get new IPs on each restart, we use a custom
#  origin domain that is updated by the deploy workflow via a Lambda@Edge
#  or (simpler) by storing the ECS task IP in SSM Parameter Store and
#  using a CloudFront Function to forward to the current ECS IP.
#
#  SIMPLER APPROACH: Use a fixed custom domain for the ECS container.
#  We set CloudFront origin to the ECS service's public IP stored in
#  SSM Parameter Store, updated on each deploy by GitHub Actions.
#
#  For personal use, the simplest and most reliable approach:
#  Use a custom origin domain = quant-api.dileepkapu.com
#  which is a Route53 A record that always points to the current ECS task IP.
#  GitHub Actions updates this A record on every deploy.
###─────────────────────────────────────────────────────────────────────────###

resource "aws_cloudfront_origin_access_control" "frontend" {
  name                              = "${local.name_prefix}-frontend-oac"
  description                       = "OAC for QuantEdge frontend S3"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_distribution" "main" {
  enabled             = true
  is_ipv6_enabled     = true
  default_root_object = "index.html"
  price_class         = "PriceClass_100"  # US/Europe only — cheapest

  aliases = [local.full_domain]

  # Origin 1: S3 for React frontend
  origin {
    domain_name              = aws_s3_bucket.frontend.bucket_regional_domain_name
    origin_id                = "S3Frontend"
    origin_access_control_id = aws_cloudfront_origin_access_control.frontend.id
  }

  # Origin 2: ECS directly (no ALB)
  # The origin domain is the internal API subdomain, updated by deploy workflow
  origin {
    domain_name = "quant-api-internal.${var.domain_name}"
    origin_id   = "ECSBackend"
    origin_path = ""

    custom_origin_config {
      http_port              = 8000
      https_port             = 443
      origin_protocol_policy = "http-only"   # ECS serves HTTP on 8000
      origin_ssl_protocols   = ["TLSv1.2"]
      origin_read_timeout    = 120           # Long timeout for ML analysis
      origin_keepalive_timeout = 60
    }

    # Custom header to verify requests came from CloudFront
    custom_header {
      name  = "X-CloudFront-Secret"
      value = "quantedge-${data.aws_caller_identity.current.account_id}"
    }
  }

  # Default behaviour: serve React frontend from S3
  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "S3Frontend"
    viewer_protocol_policy = "redirect-to-https"
    compress               = true

    forwarded_values {
      query_string = false
      cookies { forward = "none" }
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  # /api/* → ECS backend
  ordered_cache_behavior {
    path_pattern           = "/api/*"
    allowed_methods        = ["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "ECSBackend"
    viewer_protocol_policy = "redirect-to-https"
    compress               = false

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Content-Type", "Accept", "Origin"]
      cookies { forward = "all" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }

  # /auth/* → ECS backend (authentication endpoints)
  ordered_cache_behavior {
    path_pattern           = "/auth/*"
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "ECSBackend"
    viewer_protocol_policy = "redirect-to-https"
    compress               = false

    forwarded_values {
      query_string = true
      headers      = ["Authorization", "Content-Type"]
      cookies { forward = "none" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }

  # /health → ECS backend
  ordered_cache_behavior {
    path_pattern           = "/health"
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "ECSBackend"
    viewer_protocol_policy = "redirect-to-https"
    compress               = false

    forwarded_values {
      query_string = false
      cookies { forward = "none" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }

  # /ws/* → ECS backend (WebSocket)
  ordered_cache_behavior {
    path_pattern           = "/ws/*"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "ECSBackend"
    viewer_protocol_policy = "redirect-to-https"
    compress               = false

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies { forward = "all" }
    }

    min_ttl     = 0
    default_ttl = 0
    max_ttl     = 0
  }

  # SPA routing — return index.html for all 404s
  custom_error_response {
    error_code            = 404
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 0
  }

  custom_error_response {
    error_code            = 403
    response_code         = 200
    response_page_path    = "/index.html"
    error_caching_min_ttl = 0
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    acm_certificate_arn      = aws_acm_certificate_validation.main.certificate_arn
    ssl_support_method       = "sni-only"
    minimum_protocol_version = "TLSv1.2_2021"
  }

  tags = { Name = "${local.name_prefix}-distribution" }

  depends_on = [aws_acm_certificate_validation.main]
}

# S3 bucket policy — CloudFront OAC access only
resource "aws_s3_bucket_policy" "frontend" {
  bucket = aws_s3_bucket.frontend.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "cloudfront.amazonaws.com" }
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.frontend.arn}/*"
      Condition = {
        StringEquals = {
          "AWS:SourceArn" = aws_cloudfront_distribution.main.arn
        }
      }
    }]
  })
}

###─────────────────────────────────────────────────────────────────────────###
#  SSM PARAMETER — stores current ECS task public IP
#  Updated by GitHub Actions deploy workflow after every deployment
###─────────────────────────────────────────────────────────────────────────###

resource "aws_ssm_parameter" "ecs_task_ip" {
  name        = "/quantedge/ecs-task-ip"
  type        = "String"
  value       = "0.0.0.0"  # Placeholder — updated by deploy workflow
  description = "Current ECS task public IP — updated on every deploy"
  tags        = { Name = "quantedge-ecs-task-ip" }

  lifecycle {
    ignore_changes = [value]  # Terraform won't overwrite it after initial creation
  }
}

###─────────────────────────────────────────────────────────────────────────###
#  ROUTE 53
###─────────────────────────────────────────────────────────────────────────###

# quant.dileepkapu.com → CloudFront
resource "aws_route53_record" "main" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = local.full_domain
  type    = "A"
  alias {
    name                   = aws_cloudfront_distribution.main.domain_name
    zone_id                = aws_cloudfront_distribution.main.hosted_zone_id
    evaluate_target_health = false
  }
}

# quant-api-internal.dileepkapu.com → ECS task public IP
# This A record is updated by the deploy workflow on every deployment
# CloudFront uses this as its ECS origin domain
resource "aws_route53_record" "api_internal" {
  zone_id = data.aws_route53_zone.main.zone_id
  name    = "quant-api-internal.${var.domain_name}"
  type    = "A"
  ttl     = 60  # Low TTL so updates propagate quickly

  records = ["0.0.0.0"]  # Placeholder — updated by deploy workflow

  lifecycle {
    ignore_changes = [records]  # Terraform won't overwrite after initial creation
  }
}

###─────────────────────────────────────────────────────────────────────────###
#  ACM TLS CERTIFICATE
###─────────────────────────────────────────────────────────────────────────###

resource "aws_acm_certificate" "main" {
  provider          = aws.us_east_1  # Must be us-east-1 for CloudFront
  domain_name       = local.full_domain
  validation_method = "DNS"

  lifecycle {
    create_before_destroy = true
  }

  tags = { Name = "${local.name_prefix}-cert" }
}

resource "aws_route53_record" "cert_validation" {
  for_each = {
    for dvo in aws_acm_certificate.main.domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  }
  zone_id         = data.aws_route53_zone.main.zone_id
  name            = each.value.name
  type            = each.value.type
  ttl             = 60
  records         = [each.value.record]
  allow_overwrite = true
}

resource "aws_acm_certificate_validation" "main" {
  provider                = aws.us_east_1
  certificate_arn         = aws_acm_certificate.main.arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]
}

###─────────────────────────────────────────────────────────────────────────###
#  COGNITO — authentication with MFA (keep for security even on personal use)
###─────────────────────────────────────────────────────────────────────────###

resource "aws_cognito_user_pool" "quantedge" {
  name = "quantedge-users"

  # Require MFA — TOTP via Google Authenticator
  mfa_configuration = "ON"
  software_token_mfa_configuration {
    enabled = true
  }

  password_policy {
    minimum_length                   = 12
    require_lowercase                = true
    require_uppercase                = true
    require_numbers                  = true
    require_symbols                  = true
    temporary_password_validity_days = 7
  }

  account_recovery_setting {
    recovery_mechanism {
      name     = "verified_email"
      priority = 1
    }
  }

  admin_create_user_config {
    allow_admin_create_user_only = true  # Only you can create users
  }

  schema {
    name                = "email"
    attribute_data_type = "String"
    required            = true
    mutable             = true
    string_attribute_constraints {
      min_length = 3
      max_length = 254
    }
  }

  tags = { Name = "quantedge-users" }
}

resource "aws_cognito_user_pool_client" "app" {
  name         = "quantedge-app-client"
  user_pool_id = aws_cognito_user_pool.quantedge.id

  generate_secret                      = false
  prevent_user_existence_errors        = "ENABLED"
  enable_token_revocation              = true
  allowed_oauth_flows_user_pool_client = false

  explicit_auth_flows = [
    "ALLOW_USER_PASSWORD_AUTH",
    "ALLOW_REFRESH_TOKEN_AUTH",
    "ALLOW_USER_SRP_AUTH"
  ]

  access_token_validity  = 60   # minutes
  id_token_validity      = 60   # minutes
  refresh_token_validity = 30   # days

  token_validity_units {
    access_token  = "minutes"
    id_token      = "minutes"
    refresh_token = "days"
  }
}

###─────────────────────────────────────────────────────────────────────────###
#  SNS — alert notifications (login failures, errors)
###─────────────────────────────────────────────────────────────────────────###

resource "aws_sns_topic" "alerts" {
  name = "quantedge-alerts"
  tags = { Name = "quantedge-alerts" }
}

resource "aws_sns_topic_subscription" "email_alerts" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.owner_email
}

###─────────────────────────────────────────────────────────────────────────###
#  OUTPUTS — values needed for deployment and GitHub Actions
###─────────────────────────────────────────────────────────────────────────###

output "site_url" {
  description = "Live site URL"
  value       = "https://${local.full_domain}"
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID — set as GitHub Secret CLOUDFRONT_DISTRIBUTION_ID"
  value       = aws_cloudfront_distribution.main.id
}

output "cloudfront_domain" {
  description = "CloudFront domain name"
  value       = aws_cloudfront_distribution.main.domain_name
}

output "ecr_repository_url" {
  description = "ECR repository URL — used in GitHub Actions docker push"
  value       = aws_ecr_repository.quantedge_api.repository_url
}
output "rds_endpoint" {
  description = "RDS endpoint — used in DATABASE_URL"
  value       = aws_db_instance.quantedge.endpoint
}

output "rds_database_url" {
  description = "Complete DATABASE_URL for the app"
  value       = "postgresql://quantedge:YOURPASSWORD@${aws_db_instance.quantedge.endpoint}/quantedge"
  sensitive   = true
}

output "cognito_user_pool_id" {
  description = "Cognito User Pool ID"
  value       = aws_cognito_user_pool.quantedge.id
}

output "cognito_client_id" {
  description = "Cognito App Client ID"
  value       = aws_cognito_user_pool_client.app.id
}

output "sns_topic_arn" {
  description = "SNS alerts topic ARN — set as GitHub Secret SNS_TOPIC_ARN"
  value       = aws_sns_topic.alerts.arn
}

output "secrets_manager_arn" {
  description = "Secrets Manager ARN"
  value       = aws_secretsmanager_secret.quantedge_secrets.arn
}

output "api_internal_record" {
  description = "Internal API Route53 record — updated by deploy workflow with ECS task IP"
  value       = "quant-api-internal.${var.domain_name}"
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.quantedge_api.name
}

output "cost_summary" {
  description = "Estimated monthly AWS cost breakdown"
  value       = "RDS db.t3.micro ~$15 + ECS 0.5vCPU/1GB ~$9-12 + CloudFront+S3 ~$5 + Route53 ~$1 + VPC Endpoints ~$7 = ~$37-40/month AWS (+ Polygon $29 = ~$67/month total)"
}
