# Deployment Guide

This guide covers deploying SimpleMem MCP Server to DigitalOcean App Platform.

## Prerequisites

- DigitalOcean account
- GitHub repository with the code
- DigitalOcean Spaces bucket configured
- OpenRouter API key

## DigitalOcean Spaces Setup

### 1. Create a Spaces Bucket

1. Go to DigitalOcean Control Panel > Spaces
2. Click "Create Spaces Bucket"
3. Select a region (e.g., `nyc3`)
4. Name your bucket (e.g., `simplemem-data`)
5. Choose "Restrict File Listing" for security

### 2. Generate Spaces Access Keys

1. Go to API > Spaces Keys
2. Click "Generate New Key"
3. Name it (e.g., `simplemem-server`)
4. Copy both the Access Key and Secret Key

## Deployment Options

### Option A: Deploy via DigitalOcean App Platform UI

1. Go to DigitalOcean App Platform
2. Click "Create App"
3. Connect your GitHub repository
4. Select the branch (usually `main`)
5. Set source directory to `MCP`
6. Choose "Dockerfile" as the build type
7. Configure environment variables (see below)
8. Deploy

### Option B: Deploy via doctl CLI

```bash
# Install doctl
brew install doctl  # macOS
# or: snap install doctl  # Linux

# Authenticate
doctl auth init

# Create app from spec
doctl apps create --spec .do/app.yaml
```

### Option C: Deploy via GitHub Actions

The included GitHub Actions workflow (`.github/workflows/deploy.yml`) automatically deploys on push to main.

Required GitHub Secrets:
- `DIGITALOCEAN_ACCESS_TOKEN`: Your DO API token
- `DO_APP_ID`: Your App Platform app ID (optional)

## Environment Variables Configuration

Set these in DigitalOcean App Platform > Your App > Settings > App-Level Environment Variables:

### Required Variables

| Variable | Type | Description |
|----------|------|-------------|
| `OPENROUTER_API_KEY` | Secret | Your OpenRouter API key |
| `S3_BUCKET` | Secret | Spaces bucket name |
| `S3_ACCESS_KEY` | Secret | Spaces access key |
| `S3_SECRET_KEY` | Secret | Spaces secret key |

### Optional Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `S3_ENDPOINT` | Plain | `https://nyc3.digitaloceanspaces.com` | Spaces endpoint |
| `S3_REGION` | Plain | `nyc3` | Spaces region |
| `SIMPLEMEM_ACCESS_KEY` | Secret | - | Client auth token |
| `PORT` | Plain | `8000` | Server port |
| `DEBUG` | Plain | `false` | Debug mode |
| `LLM_MODEL` | Plain | `openai/gpt-4.1-mini` | LLM model |
| `EMBEDDING_MODEL` | Plain | `qwen/qwen3-embedding-4b` | Embedding model |

### Setting Secrets via doctl

```bash
# Set a secret
doctl apps update APP_ID --spec - <<EOF
envs:
  - key: OPENROUTER_API_KEY
    value: sk-or-your-key
    scope: RUN_TIME
    type: SECRET
EOF
```

## Post-Deployment Verification

### 1. Health Check

```bash
curl https://your-app.ondigitalocean.app/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "s3_status": "connected"
}
```

### 2. Server Info

```bash
curl https://your-app.ondigitalocean.app/api/server/info
```

### 3. MCP Initialize Test

```bash
curl -X POST https://your-app.ondigitalocean.app/mcp \
  -H "Authorization: Bearer YOUR_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'
```

## Connecting Claude Desktop to Deployed Server

Update your Claude Desktop configuration (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "simplemem": {
      "url": "https://your-app.ondigitalocean.app/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_ACCESS_KEY"
      }
    }
  }
}
```

Replace:
- `your-app.ondigitalocean.app` with your actual App Platform URL
- `YOUR_ACCESS_KEY` with your `SIMPLEMEM_ACCESS_KEY` value

## Scaling and Instance Configuration

### Instance Sizes

| Size | Memory | CPU | Monthly Cost |
|------|--------|-----|--------------|
| `basic-xxs` | 256 MB | Shared | ~$5 |
| `basic-xs` | 512 MB | Shared | ~$10 |
| `basic-s` | 1 GB | Shared | ~$20 |
| `basic-m` | 2 GB | Shared | ~$40 |
| `professional-xs` | 1 GB | 1 vCPU | ~$25 |
| `professional-s` | 2 GB | 1 vCPU | ~$50 |

Recommended starting size: `basic-xs` or `basic-s`

### Scaling Up

Via App Platform UI:
1. Go to Settings > Resources
2. Select new instance size
3. Save and deploy

Via doctl:
```bash
doctl apps update APP_ID --spec - <<EOF
services:
  - name: mcp-server
    instance_size_slug: basic-s
EOF
```

## Monitoring

### App Platform Logs

```bash
# Via CLI
doctl apps logs APP_ID --type run

# Or view in DigitalOcean Console
```

### Health Checks

App Platform automatically monitors the `/api/health` endpoint and restarts the container if health checks fail.

### Alerts

Set up alerts in DigitalOcean Console:
1. Go to your app > Insights
2. Configure alerts for CPU, memory, or failed deployments

## Troubleshooting

### Deployment Failures

1. Check build logs in App Platform console
2. Verify Dockerfile builds locally:
   ```bash
   docker build -t simplemem-mcp ./MCP
   ```

### S3 Connection Errors

1. Verify Spaces credentials are correct
2. Check endpoint matches your bucket's region
3. Ensure bucket name is correct (no URL, just the name)

### Authentication Issues

1. If `SIMPLEMEM_ACCESS_KEY` is set, ensure clients include:
   ```
   Authorization: Bearer your-access-key
   ```
2. If auth should be disabled, leave `SIMPLEMEM_ACCESS_KEY` unset

### Out of Memory

1. Upgrade instance size
2. Check for memory leaks in logs
3. Monitor concurrent connections

## Security Recommendations

1. **Always set `SIMPLEMEM_ACCESS_KEY`** in production
2. Use a strong random value: `openssl rand -hex 32`
3. Enable HTTPS (App Platform provides this automatically)
4. Keep Spaces bucket private (restrict file listing)
5. Rotate credentials periodically

## Cost Optimization

1. Start with smallest instance that works
2. Use auto-deploy only on `main` branch
3. Monitor usage and scale as needed
4. Consider reserved instances for steady workloads

## Updating the Deployment

### Manual Update

```bash
# Push changes to main branch
git push origin main

# App Platform will auto-deploy if configured
```

### Rollback

Via App Platform UI:
1. Go to Activity tab
2. Find previous successful deployment
3. Click "Rollback to this deployment"

Via doctl:
```bash
doctl apps list-deployments APP_ID
doctl apps create-deployment APP_ID --wait
```
