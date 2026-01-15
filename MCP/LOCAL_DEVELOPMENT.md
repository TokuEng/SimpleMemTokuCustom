# Local Development Guide

This guide covers how to set up and run SimpleMem MCP Server locally for development.

## Prerequisites

- Python 3.11+
- DigitalOcean Spaces bucket (or S3-compatible storage)
- OpenRouter API key

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd kabul-v1/MCP
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env` with your values:

```bash
# Required
OPENROUTER_API_KEY=sk-or-your-key-here
S3_BUCKET=your-bucket-name
S3_ACCESS_KEY=your-spaces-access-key
S3_SECRET_KEY=your-spaces-secret-key

# Optional - uncomment to enable client authentication
# SIMPLEMEM_ACCESS_KEY=your-secure-random-string
```

### 5. Run the Server

```bash
python run.py --reload
```

The server will start at `http://localhost:8000`.

## Available Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /mcp` | MCP Streamable HTTP endpoint |
| `GET /api/health` | Health check with S3 connectivity |
| `GET /api/server/info` | Server information and capabilities |

## Testing the Server

### Health Check

```bash
curl http://localhost:8000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "s3_status": "connected"
}
```

### Server Info

```bash
curl http://localhost:8000/api/server/info
```

### MCP Initialize

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0.0"}}}'
```

If authentication is enabled, add the Authorization header:
```bash
curl -X POST http://localhost:8000/mcp \
  -H "Authorization: Bearer YOUR_ACCESS_KEY" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '...'
```

## Using with Claude Desktop

Add to your Claude Desktop configuration (`~/.claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "simplemem": {
      "command": "python",
      "args": ["/path/to/kabul-v1/MCP/run.py"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-your-key",
        "S3_BUCKET": "your-bucket",
        "S3_ACCESS_KEY": "your-access-key",
        "S3_SECRET_KEY": "your-secret-key"
      }
    }
  }
}
```

Or if running as an HTTP server:

```json
{
  "mcpServers": {
    "simplemem": {
      "url": "http://localhost:8000/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_ACCESS_KEY"
      }
    }
  }
}
```

## Development Tips

### Auto-Reload

Use `--reload` flag for automatic server restart on code changes:

```bash
python run.py --reload
```

### Debug Mode

Enable debug mode in `.env`:

```bash
DEBUG=true
```

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
pip install ruff
ruff check . --fix
ruff format .
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENROUTER_API_KEY` | Yes | - | OpenRouter API key for LLM/embeddings |
| `S3_BUCKET` | Yes | - | DigitalOcean Spaces bucket name |
| `S3_ACCESS_KEY` | Yes | - | Spaces access key |
| `S3_SECRET_KEY` | Yes | - | Spaces secret key |
| `S3_ENDPOINT` | No | `https://nyc3.digitaloceanspaces.com` | S3/Spaces endpoint URL |
| `S3_REGION` | No | `nyc3` | S3/Spaces region |
| `SIMPLEMEM_ACCESS_KEY` | No | - | Client auth token (disabled if not set) |
| `PORT` | No | `8000` | Server port |
| `DEBUG` | No | `false` | Enable debug mode |
| `LLM_MODEL` | No | `openai/gpt-4.1-mini` | LLM model for processing |
| `EMBEDDING_MODEL` | No | `qwen/qwen3-embedding-4b` | Embedding model |

## Troubleshooting

### S3 Connection Issues

1. Verify your Spaces credentials are correct
2. Check that the bucket exists and is accessible
3. Ensure the endpoint URL matches your Spaces region

### Authentication Errors

If using `SIMPLEMEM_ACCESS_KEY`, ensure the client includes the correct bearer token:
```
Authorization: Bearer your-access-key
```

### Memory Issues

If encountering memory issues with large datasets:
1. Check available system memory
2. Consider adjusting batch sizes in retriever settings
3. Monitor LanceDB table size

## Project Structure

```
MCP/
├── config/
│   └── settings.py          # Configuration management
├── server/
│   ├── auth/
│   │   ├── models.py        # Data models (MemoryEntry, Dialogue)
│   │   └── token_manager.py # SimpleAuthManager
│   ├── core/
│   │   ├── memory_builder.py # Memory processing pipeline
│   │   └── retriever.py     # Adaptive retrieval system
│   ├── database/
│   │   └── vector_store.py  # SingleTenantVectorStore with S3
│   ├── http_server.py       # FastAPI application
│   └── mcp_handler.py       # MCP protocol handler
├── run.py                   # Entry point
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container configuration
└── .env.example             # Environment template
```
