# Complete Deployment Guide: Frontend + Backend to AWS

This guide covers deploying both your Next.js frontend and FastAPI backend to AWS.

## Architecture Overview

```
┌─────────────────┐         ┌──────────────────┐
│  AWS Amplify    │         │ AWS Elastic      │
│  (Frontend)     │────────▶│ Beanstalk        │
│  Next.js App    │  HTTPS  │ (Backend API)    │
└─────────────────┘         └──────────────────┘
                                      │
                                      ▼
                            ┌──────────────────┐
                            │  AWS RDS         │
                            │  (Database)      │
                            └──────────────────┘
```

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **Git Repository** (GitHub, GitLab, Bitbucket, or AWS CodeCommit)
3. **AWS CLI** installed: `pip install awscli`
4. **EB CLI** installed: `pip install awsebcli`
5. **Database** (MySQL/PostgreSQL) - can be RDS or external

## Part 1: Deploy Backend (FastAPI)

### Step 1: Install Required Tools

```bash
pip install awscli awsebcli
aws configure  # Enter your AWS credentials
```

### Step 2: Initialize Elastic Beanstalk

```bash
cd backend
eb init
```

**Configuration:**
- Region: Choose your region
- Application: `chatbot-backend`
- Platform: `Python`
- Platform version: `Python 3.11` or `3.12`
- SSH: `Yes`

### Step 3: Create Environment

```bash
eb create chatbot-backend-prod
```

Wait 5-10 minutes for the environment to be created.

### Step 4: Set Environment Variables

```bash
eb setenv \
  JWT_SECRET="your-strong-secret-key-minimum-32-characters" \
  FRONTEND_ORIGIN="https://main.xxxxx.amplifyapp.com" \
  MYSQL_HOST="your-database.rds.amazonaws.com" \
  MYSQL_USER="your-db-username" \
  MYSQL_PASSWORD="your-db-password" \
  MYSQL_DATABASE="your-database-name" \
  MYSQL_PORT="3306" \
  ENCRYPTION_KEY="your-32-character-encryption-key" \
  GOOGLE_API_KEY="your-google-gemini-api-key"
```

**Note:** You'll update `FRONTEND_ORIGIN` after deploying the frontend.

### Step 5: Get Backend URL

```bash
eb status
```

Note the CNAME. Your API base URL will be:
`http://your-cname.elasticbeanstalk.com/api`

**Important:** For production, set up HTTPS (see backend DEPLOYMENT.md for details).

---

## Part 2: Deploy Frontend (Next.js)

### Step 1: Push Code to Git

Ensure your code is pushed to your Git repository.

### Step 2: Access AWS Amplify Console

1. Go to [AWS Amplify Console](https://console.aws.amazon.com/amplify/)
2. Click **"New app"** → **"Host web app"**
3. Connect your Git repository
4. Select the repository and branch

### Step 3: Configure Build Settings

Amplify should auto-detect your Next.js app. The `amplify.yml` file is already configured.

### Step 4: Set Environment Variables

In Amplify console, add:

- **`NEXT_PUBLIC_API_BASE`**: `https://your-cname.elasticbeanstalk.com/api`
  - Replace with your actual backend URL from Step 5 above
  - Use `http://` if HTTPS isn't configured yet (not recommended for production)

### Step 5: Deploy

Click **"Save and deploy"**. Wait for the build to complete.

### Step 6: Get Frontend URL

After deployment, you'll get an Amplify URL like:
`https://main.xxxxx.amplifyapp.com`

---

## Part 3: Connect Frontend and Backend

### Step 1: Update Backend CORS

Update the backend's `FRONTEND_ORIGIN` environment variable:

```bash
eb setenv FRONTEND_ORIGIN="https://main.xxxxx.amplifyapp.com,http://localhost:3000"
```

Replace `main.xxxxx.amplifyapp.com` with your actual Amplify URL.

### Step 2: Verify Connection

1. Open your Amplify app URL in a browser
2. Try logging in or making an API call
3. Check browser console for any CORS errors
4. Check backend logs: `eb logs`

---

## Part 4: Database Setup (If Needed)

### Option A: AWS RDS (Recommended)

1. Go to [AWS RDS Console](https://console.aws.amazon.com/rds/)
2. Click **"Create database"**
3. Choose **MySQL** or **PostgreSQL**
4. Select **Free tier** (for testing) or production configuration
5. Configure:
   - Master username and password
   - Database name
   - Instance size
6. **Important:** Configure security group to allow connections from your Elastic Beanstalk security group
7. Note the endpoint URL (e.g., `your-db.xxxxx.rds.amazonaws.com`)

### Option B: External Database

If using an external database, ensure:
- It's accessible from AWS
- Security groups/firewall allow connections from AWS
- Use the external database host in `MYSQL_HOST`

---

## Environment Variables Summary

### Backend (Elastic Beanstalk)

| Variable | Description | Example |
|----------|-------------|---------|
| `JWT_SECRET` | Secret for JWT tokens | `your-secret-key` |
| `FRONTEND_ORIGIN` | Allowed frontend origins | `https://app.amplifyapp.com` |
| `MYSQL_HOST` | Database host | `db.rds.amazonaws.com` |
| `MYSQL_USER` | Database user | `admin` |
| `MYSQL_PASSWORD` | Database password | `password123` |
| `MYSQL_DATABASE` | Database name | `chatbot_db` |
| `MYSQL_PORT` | Database port | `3306` |
| `ENCRYPTION_KEY` | Encryption key | `your-32-char-key` |
| `GOOGLE_API_KEY` | Google Gemini API key | `your-api-key` |

### Frontend (Amplify)

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_BASE` | Backend API URL | `https://api.elasticbeanstalk.com/api` |

---

## Testing the Deployment

### 1. Test Frontend

- Visit your Amplify URL
- Verify the app loads correctly
- Check browser console for errors

### 2. Test Backend

```bash
# Check backend health
curl http://your-backend.elasticbeanstalk.com/api/health

# Or in browser
# http://your-backend.elasticbeanstalk.com/docs (FastAPI docs)
```

### 3. Test Integration

- Try logging in from the frontend
- Make API calls
- Check for CORS errors
- Verify data flows correctly

---

## Updating Deployments

### Update Backend

```bash
cd backend
# Make your changes
git add .
git commit -m "Your changes"
eb deploy
```

### Update Frontend

```bash
cd frontend
# Make your changes
git add .
git commit -m "Your changes"
git push
# Amplify will automatically deploy
```

---

## HTTPS Setup (Production)

### Backend HTTPS

1. Request a certificate in AWS Certificate Manager (ACM)
2. Configure Elastic Beanstalk load balancer to use HTTPS
3. Update `FRONTEND_ORIGIN` to use `https://`
4. Update frontend `NEXT_PUBLIC_API_BASE` to use `https://`

### Frontend HTTPS

Amplify provides HTTPS automatically. For custom domain:
1. Add domain in Amplify console
2. Configure DNS records
3. SSL certificate is managed automatically

---

## Monitoring and Logs

### Backend Logs

```bash
eb logs          # Download logs
eb logs --stream # Stream logs in real-time
eb ssh           # SSH into instance
```

### Frontend Logs

- View in AWS Amplify Console → App → Build history → Build logs

### CloudWatch

Both services integrate with CloudWatch for:
- Application metrics
- Error tracking
- Performance monitoring

---

## Cost Estimation

### Free Tier (First 12 Months)

- **Elastic Beanstalk**: 750 hours/month of t2.micro
- **RDS**: 750 hours/month of db.t2.micro
- **Amplify**: 1000 build minutes/month, 15 GB storage, 5 GB served/month

### After Free Tier (Estimated Monthly)

- **Elastic Beanstalk**: ~$15-30 (t3.small instance)
- **RDS**: ~$15-30 (db.t3.micro)
- **Amplify**: Free tier usually sufficient for small apps
- **Data Transfer**: ~$0.09/GB (first 1GB free)

**Total**: ~$30-60/month for a small application

---

## Troubleshooting

### Backend Issues

**Application won't start:**
- Check logs: `eb logs`
- Verify environment variables: `eb printenv`
- Check Python version compatibility

**Database connection fails:**
- Verify RDS security group allows EB security group
- Check database credentials
- Verify endpoint URL

**CORS errors:**
- Ensure `FRONTEND_ORIGIN` includes your Amplify URL
- Check backend logs for CORS configuration

### Frontend Issues

**Build fails:**
- Check build logs in Amplify console
- Verify all dependencies in `package.json`
- Check Node.js version compatibility

**API calls fail:**
- Verify `NEXT_PUBLIC_API_BASE` is correct
- Check CORS settings on backend
- Verify backend is running: `eb status`

**404 errors:**
- Check Next.js routing configuration
- Verify middleware is working correctly

---

## Security Checklist

- [ ] Use HTTPS for both frontend and backend
- [ ] Strong JWT secret (minimum 32 characters)
- [ ] Strong database passwords
- [ ] Restrict database access to backend only
- [ ] Use environment variables, never commit secrets
- [ ] Configure CORS to only allow your frontend domain
- [ ] Enable database encryption at rest (RDS)
- [ ] Regular security updates
- [ ] Monitor CloudWatch for suspicious activity

---

## Next Steps

1. ✅ Deploy backend to Elastic Beanstalk
2. ✅ Deploy frontend to Amplify
3. ✅ Connect frontend and backend
4. ⬜ Set up custom domains
5. ⬜ Configure HTTPS
6. ⬜ Set up monitoring and alerts
7. ⬜ Configure auto-scaling (if needed)
8. ⬜ Set up backup strategy for database

---

## Additional Resources

- [Backend Deployment Guide](./backend/DEPLOYMENT.md) - Detailed backend deployment
- [Backend Quick Start](./backend/QUICK_START.md) - Quick backend deployment
- [Frontend Deployment Guide](./frontend/DEPLOYMENT.md) - Detailed frontend deployment
- [AWS Elastic Beanstalk Docs](https://docs.aws.amazon.com/elasticbeanstalk/)
- [AWS Amplify Docs](https://docs.aws.amazon.com/amplify/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)

---

## Support

If you encounter issues:
1. Check the logs (backend: `eb logs`, frontend: Amplify console)
2. Review the troubleshooting sections in individual deployment guides
3. Check AWS service health status
4. Review CloudWatch metrics for insights

