# Deploying Next.js Frontend to AWS Amplify

This guide will walk you through deploying your Next.js application to AWS Amplify.

## Prerequisites

1. An AWS account
2. Your Next.js application code in a Git repository (GitHub, GitLab, Bitbucket, or AWS CodeCommit)
3. Your backend API URL (where your FastAPI backend is hosted)

## Step-by-Step Deployment

### 1. Prepare Your Repository

Ensure your code is pushed to your Git repository (GitHub, GitLab, Bitbucket, or AWS CodeCommit).

### 2. Access AWS Amplify Console

1. Go to [AWS Amplify Console](https://console.aws.amazon.com/amplify/)
2. Sign in to your AWS account
3. Click **"New app"** → **"Host web app"**

### 3. Connect Your Repository

1. Choose your Git provider (GitHub, GitLab, Bitbucket, or AWS CodeCommit)
2. Authorize AWS Amplify to access your repository
3. Select the repository containing your Next.js frontend
4. Select the branch you want to deploy (usually `main` or `master`)

### 4. Configure Build Settings

AWS Amplify should automatically detect your Next.js app. The build settings should look like this:

**Build image:** `Amazon Linux 2023` (or latest available)

**Build settings:**
```yaml
version: 1
frontend:
  phases:
    preBuild:
      commands:
        - npm ci
    build:
      commands:
        - npm run build
  artifacts:
    baseDirectory: .next
    files:
      - '**/*'
  cache:
    paths:
      - node_modules/**/*
      - .next/cache/**/*
```

**Note:** The `amplify.yml` file in your frontend directory already contains these settings, so Amplify should auto-detect them.

### 5. Configure Environment Variables

In the Amplify console, go to **"Environment variables"** and add:

- **`NEXT_PUBLIC_API_BASE`**: Your backend API URL (e.g., `https://your-api-domain.com/api`)
  - This should be the full URL where your FastAPI backend is hosted
  - Example: `https://api.yourdomain.com/api` or `https://your-backend.elasticbeanstalk.com/api`

**Important:** 
- Environment variables prefixed with `NEXT_PUBLIC_` are exposed to the browser
- Make sure your backend CORS settings allow requests from your Amplify domain
- Your backend should have `FRONTEND_ORIGIN` set to include your Amplify app URL

### 6. Review and Deploy

1. Review your build settings
2. Click **"Save and deploy"**
3. AWS Amplify will:
   - Clone your repository
   - Install dependencies
   - Build your Next.js application
   - Deploy it to a CDN

### 7. Update Backend CORS Settings

After deployment, you'll get an Amplify app URL (e.g., `https://main.xxxxx.amplifyapp.com`). 

**Update your backend environment variables:**
- Add your Amplify URL to the `FRONTEND_ORIGIN` environment variable in your backend
- Example: `FRONTEND_ORIGIN=https://main.xxxxx.amplifyapp.com,http://localhost:3000`
- Restart your backend server

### 8. Custom Domain (Optional)

1. In the Amplify console, go to **"Domain management"**
2. Click **"Add domain"**
3. Enter your domain name
4. Follow the DNS configuration instructions
5. AWS will provide you with DNS records to add to your domain registrar

## Important Notes

### Next.js 15 and AWS Amplify

- AWS Amplify supports Next.js 15 with Server-Side Rendering (SSR)
- The `amplify.yml` configuration handles the build process automatically
- Static pages are served from CDN, while dynamic routes use SSR

### Environment Variables

- All environment variables must be set in the Amplify console
- Variables prefixed with `NEXT_PUBLIC_` are available in the browser
- Other variables are only available during build time

### Build Optimization

- Amplify automatically caches `node_modules` and `.next/cache` between builds
- This speeds up subsequent deployments

### Troubleshooting

**Build fails:**
- Check the build logs in the Amplify console
- Ensure all dependencies are in `package.json`
- Verify Node.js version compatibility (Amplify uses Node.js 18+ by default)

**API calls fail:**
- Verify `NEXT_PUBLIC_API_BASE` is set correctly
- Check CORS settings on your backend
- Ensure your backend allows requests from the Amplify domain

**404 errors on routes:**
- Next.js middleware and dynamic routes work with Amplify
- If you have issues, check the `next.config.mjs` for any special configurations

## Continuous Deployment

AWS Amplify automatically deploys when you push to your connected branch. You can:
- Set up branch-based deployments
- Configure preview deployments for pull requests
- Set up custom build notifications

## Cost

- AWS Amplify offers a free tier with generous limits
- Free tier includes: 1000 build minutes/month, 15 GB storage, 5 GB served/month
- Check [AWS Amplify Pricing](https://aws.amazon.com/amplify/pricing/) for current pricing

## Additional Resources

- [AWS Amplify Documentation](https://docs.aws.amazon.com/amplify/)
- [Next.js Deployment Documentation](https://nextjs.org/docs/deployment)
- [Amplify Build Settings Reference](https://docs.aws.amazon.com/amplify/latest/userguide/build-settings.html)

