# Quick Start: Deploy Backend to AWS Elastic Beanstalk

## Prerequisites

1. Install AWS CLI: `pip install awscli`
2. Install EB CLI: `pip install awsebcli`
3. Configure AWS credentials: `aws configure`

## Step-by-Step Deployment

### 1. Navigate to Backend Directory

```bash
cd backend
```

### 2. Initialize Elastic Beanstalk

```bash
eb init
```

**Follow prompts:**
- Region: Choose your region (e.g., `us-east-1`)
- Application name: `chatbot-backend`
- Platform: `Python`
- Platform version: `Python 3.11` or `3.12`
- SSH: `Yes` (recommended for debugging)

### 3. Create Environment

```bash
eb create chatbot-backend-prod
```

This takes 5-10 minutes. Wait for it to complete.

### 4. Set Environment Variables

```bash
eb setenv \
  JWT_SECRET="your-strong-secret-key-min-32-chars" \
  FRONTEND_ORIGIN="https://your-amplify-app.amplifyapp.com" \
  MYSQL_HOST="your-database.rds.amazonaws.com" \
  MYSQL_USER="your-db-user" \
  MYSQL_PASSWORD="your-db-password" \
  MYSQL_DATABASE="your-database-name" \
  MYSQL_PORT="3306" \
  ENCRYPTION_KEY="your-encryption-key-32-chars" \
  GOOGLE_API_KEY="your-google-gemini-api-key"
```

**Important:** Replace all placeholder values with your actual credentials.

### 5. Get Your Backend URL

```bash
eb status
```

Look for the "CNAME" value. Your API will be at:
`http://your-cname.elasticbeanstalk.com/api`

### 6. Update Frontend

In AWS Amplify console, update the `NEXT_PUBLIC_API_BASE` environment variable to:
`https://your-cname.elasticbeanstalk.com/api`

(Note: You may need to set up HTTPS separately - see full deployment guide)

### 7. Deploy Updates

Whenever you make code changes:

```bash
eb deploy
```

## Useful Commands

```bash
# View logs
eb logs

# SSH into instance
eb ssh

# Check environment health
eb health

# Open in browser
eb open

# List environments
eb list

# Terminate environment (careful!)
eb terminate chatbot-backend-prod
```

## Troubleshooting

**Build fails:**
- Check logs: `eb logs`
- Verify all dependencies in `requirements.txt`
- Check Python version compatibility

**Application errors:**
- SSH into instance: `eb ssh`
- Check application logs: `eb logs`
- Verify environment variables: `eb printenv`

**Database connection issues:**
- Verify RDS security group allows connections from EB security group
- Check database credentials
- Verify database endpoint URL

## Next Steps

1. Set up RDS database (if not already done)
2. Configure HTTPS (optional but recommended)
3. Set up auto-scaling (optional)
4. Configure custom domain (optional)

For detailed information, see `DEPLOYMENT.md`.

