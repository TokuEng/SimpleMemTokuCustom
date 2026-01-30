# SimpleMem MCP Server - Single-Tenant Migration

## Overview

This PR migrates SimpleMem from a **multi-tenant architecture** (per-user API keys, JWT authentication, local storage) to a **single-tenant architecture** (server-side API key, S3 storage, optional simple auth) for deployment on DigitalOcean App Platform.

---

## Deployed Server

| Item | Value |
|------|-------|
| **URL** | `https://simplemem-mcp-server-q56ck.ondigitalocean.app` |
| **MCP Endpoint** | `POST /mcp` |
| **Memory Viewer** | `/memories` |
| **Health Check** | `/api/health` |
| **Memories API** | `/api/memories` |
| **Protocol** | MCP 2025-03-26 (Streamable HTTP) |

---

## What SimpleMem Does

SimpleMem is an **advanced lifelong memory system for LLM agents**. It provides:

- **Semantic Lossless Compression**: Converts conversations into self-contained facts (no pronouns, absolute timestamps)
- **Coreference Resolution**: "He said he'll meet her" → "Bob said Bob will meet Alice"
- **Temporal Anchoring**: "tomorrow at 3pm" → "2025-01-16 at 15:00"
- **Hybrid Retrieval**: Combines semantic search, keyword matching, and metadata filtering
- **Entity Extraction**: Automatically extracts persons, locations, entities, and topics

---

## Architecture Changes

### Before (Multi-Tenant)
```
User → Register with OpenRouter API key → Get JWT token
     → Use JWT token to authenticate
     → Per-user LanceDB table (user_xxx)
     → Local file storage
```

### After (Single-Tenant)
```
Server has OpenRouter API key (environment variable)
     → Optional Bearer token authentication
     → Single shared LanceDB table (memories)
     → S3/DigitalOcean Spaces storage
```

---

## Key Files Modified

### Configuration (`MCP/config/settings.py`)
- Added S3 configuration: `S3_BUCKET`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_ENDPOINT`, `S3_REGION`
- Added `SIMPLEMEM_ACCESS_KEY` for optional client authentication
- Server-side `OPENROUTER_API_KEY` (no longer per-user)
- Added `get_s3_storage_options()` method

### Vector Store (`MCP/server/database/vector_store.py`)
- Created `SingleTenantVectorStore` class
- Connects to LanceDB with S3 path: `s3://{bucket}/lancedb`
- Single fixed table name: `memories`
- Removed `table_name` parameter from all methods

### Authentication (`MCP/server/auth/token_manager.py`)
- Created `SimpleAuthManager` class (replaces JWT `TokenManager`)
- Validates bearer token against `SIMPLEMEM_ACCESS_KEY`
- If no key set, auth is disabled (allow all)

### HTTP Server (`MCP/server/http_server.py`)
- Removed registration endpoints (`/api/auth/register`, `/api/auth/verify`, `/api/auth/refresh`)
- Single global `OpenRouterClient` and `SingleTenantVectorStore`
- Simplified `verify_bearer_token()` to check against `SIMPLEMEM_ACCESS_KEY`
- Added resilient session management (auto-creates sessions when expired)
- Added Memory Viewer dashboard at `/memories`
- Added JSON API at `/api/memories`

### MCP Handler (`MCP/server/mcp_handler.py`)
- Removed user context from constructor
- Uses single vector store and OpenRouter client
- Updated instructions for better tool triggering in Claude Code

---

## MCP Tools Available

| Tool | Description |
|------|-------------|
| `memory_add` | Store a single dialogue/fact with semantic processing |
| `memory_add_batch` | Store multiple dialogues at once |
| `memory_query` | Ask questions about stored memories (returns AI-synthesized answer) |
| `memory_retrieve` | Get raw memory entries with full metadata |
| `memory_stats` | Get statistics (count, storage mode) |
| `memory_clear` | Delete all memories (irreversible) |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | Yes | OpenRouter API key for LLM calls |
| `S3_BUCKET` | Yes | DigitalOcean Spaces bucket name |
| `S3_ACCESS_KEY` | Yes | Spaces access key (must be global, not bucket-scoped) |
| `S3_SECRET_KEY` | Yes | Spaces secret key |
| `S3_ENDPOINT` | No | Default: `https://nyc3.digitaloceanspaces.com` |
| `S3_REGION` | No | Default: `nyc3` |
| `SIMPLEMEM_ACCESS_KEY` | No | Client auth token (if unset, auth disabled) |
| `PORT` | No | Default: `8000` |
| `LLM_MODEL` | No | Default: `openai/gpt-4.1-mini` |
| `EMBEDDING_MODEL` | No | Default: `qwen/qwen3-embedding-4b` |

---

## Connecting Claude Code to SimpleMem

### Add MCP Server
```bash
claude mcp add --transport http simplemem https://simplemem-mcp-server-q56ck.ondigitalocean.app/mcp
```

### Verify Connection
```
/mcp
```
Should show: `simplemem · ✓ connected · 6 tools`

### Test Commands
```
Remember that my favorite color is blue
What's my favorite color?
How many memories do you have stored?
Show me all stored memories
```

---

## Memory Viewer Dashboard

Access at: `https://simplemem-mcp-server-q56ck.ondigitalocean.app/memories`

Features:
- **Stats Bar**: Total memories, S3 bucket, connection status
- **Memory Cards**: Each memory with full metadata (timestamp, persons, location, entities, topic)
- **Search**: Real-time filtering across all memories
- **Clear All**: Delete all memories with confirmation modal
- **Responsive**: Works on mobile devices

---

## API Endpoints

### Health Check
```bash
curl https://simplemem-mcp-server-q56ck.ondigitalocean.app/api/health
```

### Get All Memories (JSON)
```bash
curl https://simplemem-mcp-server-q56ck.ondigitalocean.app/api/memories
```

### MCP Initialize
```bash
curl -X POST https://simplemem-mcp-server-q56ck.ondigitalocean.app/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  -d '{"jsonrpc":"2.0","method":"initialize","id":1,"params":{}}'
```

---

## Session Management

The server implements resilient session management:

1. **Initialize Request**: Creates new session, returns `Mcp-Session-Id` header
2. **Subsequent Requests**: Use `Mcp-Session-Id` header
3. **Session Expiry**: 30 minutes of inactivity
4. **Auto-Recovery**: If session not found, automatically creates new one (no errors)

This ensures Claude Code can always make tool calls even after server restarts or session timeouts.

---

## Deployment

### DigitalOcean App Platform

1. Connect GitHub repo
2. Select `MCP` as source directory
3. Configure environment variables (secrets in UI)
4. Deploy

### Configuration File
See `MCP/.do/app.yaml` for full deployment configuration.

---

## Files Structure

```
MCP/
├── .do/
│   └── app.yaml              # DigitalOcean deployment config
├── config/
│   └── settings.py           # S3 and server configuration
├── server/
│   ├── auth/
│   │   ├── __init__.py
│   │   ├── models.py         # MemoryEntry, Dialogue models
│   │   └── token_manager.py  # SimpleAuthManager
│   ├── core/
│   │   ├── memory_builder.py # Dialogue processing
│   │   ├── retriever.py      # Hybrid search
│   │   └── answer_generator.py
│   ├── database/
│   │   ├── __init__.py
│   │   └── vector_store.py   # SingleTenantVectorStore
│   ├── integrations/
│   │   └── openrouter.py     # OpenRouter client
│   ├── http_server.py        # FastAPI server + Memory Viewer
│   └── mcp_handler.py        # MCP protocol handler
├── Dockerfile
├── requirements.txt
└── run.py                    # Entry point
```

---

## Commits in This PR

1. **Single-tenant migration** - Core architecture changes
2. **Fix import errors** - Updated `__init__.py` files for new classes
3. **Replace frontend** - Simple status page instead of registration UI
4. **Session resilience** - Auto-create sessions when expired
5. **Improve MCP instructions** - Better tool triggering for Claude Code
6. **Add Memory Viewer** - Dashboard at `/memories`
7. **Fix stale stats** - Use actual entry count, not cached

---

## Testing Checklist

- [x] Health endpoint returns `healthy`
- [x] S3 connection successful
- [x] MCP initialize works
- [x] `memory_add` stores data
- [x] `memory_query` retrieves and answers
- [x] `memory_retrieve` shows raw entries
- [x] `memory_stats` shows correct count
- [x] `memory_clear` deletes all
- [x] Memory Viewer displays entries
- [x] Memory Viewer search works
- [x] Memory Viewer clear works
- [x] Claude Code can connect
- [x] Session auto-recovery works

---

## Known Issues & Notes

1. **S3 Access Keys**: Must use global Spaces access keys, not bucket-scoped keys (LanceDB limitation)
2. **Session Expiry**: Sessions expire after 30 minutes of inactivity, but auto-recover
3. **OAuth 404s in logs**: Expected - Claude Code checks for OAuth endpoints that don't exist
4. **Eventual Consistency**: S3 may have slight delays in reflecting changes

---

## Original Multi-Tenant Code

The original multi-tenant implementation is preserved:
- As comments in relevant files (for reference)
- In the `multi-tenant-backup` branch (full implementation)
