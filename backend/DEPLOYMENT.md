# Deploying FastAPI Backend to AWS

This guide covers deploying your FastAPI backend to AWS. You have two main options:

## Option 1: AWS Elastic Beanstalk (Recommended - Easiest)

AWS Elastic Beanstalk is the easiest way to deploy Python applications to AWS. It handles infrastructure, scaling, and load balancing automatically.

### Prerequisites

1. AWS Account
2. AWS CLI installed and configured
3. EB CLI (Elastic Beanstalk CLI) installed
4. Your code in a Git repository

### Step 1: Install EB CLI

```bash
pip install awsebcli
```

### Step 2: Initialize Elastic Beanstalk

Navigate to your backend directory:

```bash
cd backend
eb init
```

Follow the prompts:
- **Select a region**: Choose your preferred AWS region
- **Select an application**: Create a new application (e.g., "chatbot-backend")
- **Select a platform**: Choose "Python"
- **Select a platform version**: Choose Python 3.11 or 3.12
- **SSH access**: Choose "Yes" if you want SSH access for debugging

### Step 3: Create Environment

```bash
eb create chatbot-backend-env
```

This will:
- Create an EC2 instance
- Set up a load balancer
- Configure security groups
- Deploy your application

**Note:** This process takes 5-10 minutes.

### Step 4: Configure Environment Variables

After the environment is created, set your environment variables:

```bash
eb setenv \
  JWT_SECRET="your-jwt-secret-key-here" \
  FRONTEND_ORIGIN="https://your-amplify-app.amplifyapp.com" \
  MYSQL_HOST="your-database-host.rds.amazonaws.com" \
  MYSQL_USER="your-db-username" \
  MYSQL_PASSWORD="your-db-password" \
  MYSQL_DATABASE="your-database-name" \
  MYSQL_PORT="3306" \
  ENCRYPTION_KEY="your-encryption-key-here" \
  GOOGLE_API_KEY="your-google-api-key"
```

**Or set them via AWS Console:**
1. Go to Elastic Beanstalk Console
2. Select your environment
3. Go to **Configuration** → **Software** → **Environment properties**
4. Add all environment variables

### Step 5: Deploy Updates

Whenever you make changes:

```bash
eb deploy
```

### Step 6: Get Your Backend URL

```bash
eb status
```

Or check the Elastic Beanstalk console. Your API will be available at:
`http://your-env.elasticbeanstalk.com`

### Step 7: Update Frontend API URL

Update your frontend's `NEXT_PUBLIC_API_BASE` environment variable in AWS Amplify to:
`https://your-env.elasticbeanstalk.com/api`

### Important Notes for Elastic Beanstalk

1. **Database**: You'll need to set up an RDS MySQL/PostgreSQL instance or use an external database
2. **HTTPS**: Elastic Beanstalk provides HTTP by default. For HTTPS:
   - Use AWS Certificate Manager (ACM) to get a certificate
   - Configure the load balancer to use HTTPS
3. **WebSocket Support**: Elastic Beanstalk supports WebSockets, but you may need to configure sticky sessions
4. **Scaling**: Elastic Beanstalk can auto-scale based on traffic

### Troubleshooting Elastic Beanstalk

**View logs:**
```bash
eb logs
```

**SSH into instance:**
```bash
eb ssh
```

**Check environment health:**
```bash
eb health
```

---

## Option 2: Docker + AWS ECS/Fargate (More Control)

This option gives you more control and is better for containerized deployments.

### Prerequisites

1. AWS Account
2. AWS CLI installed
3. Docker installed locally
4. ECR (Elastic Container Registry) access

### Step 1: Build Docker Image

```bash
cd backend
docker build -t chatbot-backend .
```

### Step 2: Test Locally

```bash
docker run -p 8000:8000 \
  -e JWT_SECRET="test-secret" \
  -e FRONTEND_ORIGIN="http://localhost:3000" \
  -e MYSQL_HOST="your-db-host" \
  -e MYSQL_USER="your-db-user" \
  -e MYSQL_PASSWORD="your-db-password" \
  -e MYSQL_DATABASE="your-db-name" \
  -e GOOGLE_API_KEY="your-api-key" \
  chatbot-backend
```

### Step 3: Create ECR Repository

```bash
aws ecr create-repository --repository-name chatbot-backend
```

### Step 4: Push Image to ECR

```bash
# Get login token
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Tag image
docker tag chatbot-backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/chatbot-backend:latest

# Push image
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/chatbot-backend:latest
```

### Step 5: Create ECS Task Definition

Create a file `task-definition.json`:

```json
{
  "family": "chatbot-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "containerDefinitions": [
    {
      "name": "chatbot-backend",
      "image": "<account-id>.dkr.ecr.us-east-1.amazonaws.com/chatbot-backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "JWT_SECRET",
          "value": "your-jwt-secret"
        },
        {
          "name": "FRONTEND_ORIGIN",
          "value": "https://your-amplify-app.amplifyapp.com"
        },
        {
          "name": "MYSQL_HOST",
          "value": "your-db-host.rds.amazonaws.com"
        },
        {
          "name": "MYSQL_USER",
          "value": "your-db-user"
        },
        {
          "name": "MYSQL_PASSWORD",
          "value": "your-db-password"
        },
        {
          "name": "MYSQL_DATABASE",
          "value": "your-db-name"
        },
        {
          "name": "MYSQL_PORT",
          "value": "3306"
        },
        {
          "name": "GOOGLE_API_KEY",
          "value": "your-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/chatbot-backend",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Step 6: Register Task Definition

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json
```

### Step 7: Create ECS Service

Use AWS Console or CLI to create an ECS service with:
- Fargate launch type
- Application Load Balancer
- VPC and subnets
- Security groups allowing port 8000

### Step 8: Set Up Application Load Balancer

1. Create an Application Load Balancer
2. Configure target group to point to your ECS service
3. Set up HTTPS listener (optional but recommended)
4. Get the load balancer DNS name

### Step 9: Update Frontend

Update your frontend's `NEXT_PUBLIC_API_BASE` to your load balancer URL.

---

## Option 3: AWS Lambda + API Gateway (Serverless)

**Note:** This option is more complex due to WebSocket support and may require significant code changes. Not recommended unless you specifically need serverless architecture.

---

## Required Environment Variables

Your backend requires these environment variables:

| Variable | Description | Example |
|----------|-------------|---------|
| `JWT_SECRET` | Secret key for JWT token signing | `your-secret-key-here` |
| `FRONTEND_ORIGIN` | Allowed frontend origins (comma-separated) | `https://app.amplifyapp.com,http://localhost:3000` |
| `MYSQL_HOST` | Database host | `your-db.rds.amazonaws.com` |
| `MYSQL_USER` | Database username | `admin` |
| `MYSQL_PASSWORD` | Database password | `your-password` |
| `MYSQL_DATABASE` | Database name | `chatbot_db` |
| `MYSQL_PORT` | Database port | `3306` or `5432` |
| `ENCRYPTION_KEY` | Encryption key for sensitive data | `your-encryption-key` |
| `GOOGLE_API_KEY` | Google Gemini API key | `your-google-api-key` |

---

## Database Setup (RDS)

If you need to set up a database on AWS:

### Create RDS MySQL Instance

1. Go to AWS RDS Console
2. Click **Create database**
3. Choose **MySQL** or **PostgreSQL**
4. Select **Free tier** (for testing) or production configuration
5. Set master username and password
6. Configure security group to allow connections from your backend
7. Note the endpoint URL

### Security Group Configuration

Your RDS security group should allow inbound connections on port 3306 (MySQL) or 5432 (PostgreSQL) from:
- Your Elastic Beanstalk security group (if using EB)
- Your ECS security group (if using ECS)
- Or your specific IP for testing

---

## Cost Estimation

### Elastic Beanstalk (Free Tier Available)
- **Free Tier**: 750 hours/month of t2.micro/t3.micro for 12 months
- **After Free Tier**: ~$15-30/month for t3.small instance
- **RDS Free Tier**: 750 hours/month of db.t2.micro for 12 months

### ECS Fargate
- **Compute**: ~$0.04/vCPU-hour + ~$0.004/GB-hour
- **Example**: 0.5 vCPU, 1GB RAM = ~$15/month (always running)

---

## Security Best Practices

1. **Never commit secrets**: Use environment variables or AWS Secrets Manager
2. **Use HTTPS**: Always use HTTPS in production
3. **Database Security**: 
   - Use RDS with encryption at rest
   - Restrict database access to backend only
   - Use strong passwords
4. **CORS**: Only allow your frontend domain in `FRONTEND_ORIGIN`
5. **JWT Secret**: Use a strong, random secret key

---

## Monitoring and Logs

### Elastic Beanstalk
- View logs: `eb logs` or AWS Console
- CloudWatch integration for metrics

### ECS
- CloudWatch Logs for container logs
- CloudWatch Metrics for service health

---

## Next Steps

1. Choose your deployment option (recommend Elastic Beanstalk for simplicity)
2. Set up your database (RDS or external)
3. Deploy your backend
4. Update frontend environment variables with backend URL
5. Test the integration

---

## Additional Resources

- [AWS Elastic Beanstalk Python Guide](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-apps.html)
- [AWS ECS Documentation](https://docs.aws.amazon.com/ecs/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [AWS RDS Documentation](https://docs.aws.amazon.com/rds/)

