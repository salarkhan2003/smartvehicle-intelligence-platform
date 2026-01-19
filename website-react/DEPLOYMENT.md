# SIGHTLINE - Vercel Deployment Guide

## Prerequisites

1. **GitHub Account**: Make sure you have a GitHub account
2. **Vercel Account**: Sign up at [vercel.com](https://vercel.com) (you can use your GitHub account)
3. **Git**: Make sure Git is installed on your system

## Step 1: Prepare Your Repository

### Option A: Create a New Repository (Recommended)

1. Go to [GitHub](https://github.com) and create a new repository
2. Name it something like `sightline-website` or `sightline-ai-platform`
3. Don't initialize with README (we'll push existing code)

### Option B: Use Existing Repository

If you already have a Git repository, make sure it's up to date.

## Step 2: Push Your Code to GitHub

Open your terminal/command prompt in the `website-react` directory and run:

```bash
# Initialize git repository (if not already done)
git init

# Add all files
git add .

# Commit your changes
git commit -m "Initial commit: SIGHTLINE AI Platform"

# Add your GitHub repository as remote (replace with your repository URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git

# Push to GitHub
git push -u origin main
```

## Step 3: Deploy to Vercel

### Method 1: Vercel Dashboard (Easiest)

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click "New Project"
3. Import your GitHub repository
4. Configure the project:
   - **Framework Preset**: Next.js
   - **Root Directory**: `./` (or leave empty)
   - **Build Command**: `npm run build`
   - **Output Directory**: `.next`
   - **Install Command**: `npm install`

5. Click "Deploy"

### Method 2: Vercel CLI

1. Install Vercel CLI:
```bash
npm install -g vercel
```

2. Login to Vercel:
```bash
vercel login
```

3. Deploy from your project directory:
```bash
cd website-react
vercel
```

4. Follow the prompts:
   - Link to existing project? **N**
   - What's your project's name? **sightline-ai-platform**
   - In which directory is your code located? **.**
   - Want to override the settings? **N**

## Step 4: Configure Domain (Optional)

1. In your Vercel dashboard, go to your project
2. Click on "Settings" → "Domains"
3. Add your custom domain if you have one
4. Follow Vercel's instructions to configure DNS

## Step 5: Environment Variables (If Needed)

If you need to add environment variables:

1. Go to your project in Vercel dashboard
2. Click "Settings" → "Environment Variables"
3. Add any required variables

## Step 6: Automatic Deployments

Once connected to GitHub, Vercel will automatically:
- Deploy when you push to the main branch
- Create preview deployments for pull requests
- Show build logs and deployment status

## Troubleshooting

### Common Issues:

1. **Build Fails**: Check the build logs in Vercel dashboard
2. **Video Not Loading**: Make sure video files are in the `public/assets/` directory
3. **404 Errors**: Check that all routes are properly configured

### Build Commands:

If you need to customize build settings:

```json
{
  "scripts": {
    "build": "next build",
    "start": "next start",
    "dev": "next dev",
    "lint": "next lint"
  }
}
```

## Performance Optimization

Your site includes:
- ✅ Automatic image optimization
- ✅ Static file caching
- ✅ CDN distribution
- ✅ Gzip compression
- ✅ Modern JavaScript bundling

## Monitoring

After deployment, you can:
- View analytics in Vercel dashboard
- Monitor performance metrics
- Check deployment logs
- Set up custom monitoring

## Support

- [Vercel Documentation](https://vercel.com/docs)
- [Next.js Documentation](https://nextjs.org/docs)
- [Vercel Community](https://github.com/vercel/vercel/discussions)

---

Your SIGHTLINE AI Platform will be live at: `https://your-project-name.vercel.app`

## Quick Deploy Button

You can also use this button for one-click deployment:

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME)