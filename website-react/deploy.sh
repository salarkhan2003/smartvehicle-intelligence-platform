#!/bin/bash

# SIGHTLINE - Quick Deployment Script
echo "🚀 Deploying SIGHTLINE AI Platform to Vercel..."

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
fi

# Add all files
echo "📦 Adding files to Git..."
git add .

# Commit changes
echo "💾 Committing changes..."
git commit -m "Deploy: SIGHTLINE AI Platform $(date)"

# Check if remote exists
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "⚠️  No Git remote found. Please add your GitHub repository:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git"
    echo "   Then run this script again."
    exit 1
fi

# Push to GitHub
echo "🔄 Pushing to GitHub..."
git push origin main

# Deploy to Vercel (if CLI is installed)
if command -v vercel &> /dev/null; then
    echo "🌐 Deploying to Vercel..."
    vercel --prod
else
    echo "✅ Code pushed to GitHub!"
    echo "🌐 Go to https://vercel.com to deploy your project"
    echo "   or install Vercel CLI: npm install -g vercel"
fi

echo "🎉 Deployment process completed!"