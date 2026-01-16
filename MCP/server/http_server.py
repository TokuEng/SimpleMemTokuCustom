"""
HTTP Server for SimpleMem MCP - Single Tenant Mode

Provides:
- MCP over Streamable HTTP (2025-03-26 spec) (/mcp)
- Health and info endpoints (/api/*)
- Static frontend for configuration (/)
- S3-backed LanceDB storage

Streamable HTTP Transport:
- Single endpoint /mcp supporting POST, GET, DELETE
- Optional authentication via Authorization: Bearer <token> header
- Session management via Mcp-Session-Id header
- Supports both JSON and SSE response formats
"""

import asyncio
import html
import json
import logging
import os
import re
import secrets
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from .auth.token_manager import SimpleAuthManager
from .database.vector_store import SingleTenantVectorStore
from .integrations.openrouter import OpenRouterClient
from .mcp_handler import MCPHandler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_settings

# Configure logger
logger = logging.getLogger(__name__)


# === Session Management ===

@dataclass
class MCPSession:
    """Represents an active MCP session (single-tenant)"""
    session_id: str
    handler: MCPHandler
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_active: datetime = field(default_factory=datetime.utcnow)
    initialized: bool = False
    # Server-to-client message queue for GET requests
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # Track active SSE streams
    active_streams: set = field(default_factory=set)
    # Event ID counter for resumability
    event_counter: int = 0

    def next_event_id(self) -> str:
        """Generate next SSE event ID"""
        self.event_counter += 1
        return f"{self.session_id}-{self.event_counter}"

    def touch(self):
        """Update last active timestamp"""
        self.last_active = datetime.utcnow()


# Session expiry time (30 minutes of inactivity)
SESSION_EXPIRY_MINUTES = 30


# === Global Instances ===

settings = get_settings()

# Single OpenRouter client using server-side API key
openrouter_client = OpenRouterClient(
    api_key=settings.openrouter_api_key,
    base_url=settings.openrouter_base_url,
    llm_model=settings.llm_model,
    embedding_model=settings.embedding_model,
)

# Single vector store with S3 backend
vector_store = SingleTenantVectorStore(
    s3_path=settings.lancedb_path,
    storage_options=settings.get_s3_storage_options(),
    table_name=settings.table_name,
    embedding_dimension=settings.embedding_dimension,
)

# Simple auth manager
auth_manager = SimpleAuthManager(settings.simplemem_access_key)

# Single global MCP handler (initialized at module load for thread safety)
mcp_handler = MCPHandler(
    openrouter_client=openrouter_client,
    vector_store=vector_store,
    settings=settings,
)

# Store active sessions by session_id
_sessions: dict[str, MCPSession] = {}

# Lock for session operations
_session_lock = asyncio.Lock()


# === Session Helper Functions ===

async def cleanup_expired_sessions():
    """Remove expired sessions"""
    async with _session_lock:
        now = datetime.utcnow()
        expired = [
            sid for sid, session in _sessions.items()
            if (now - session.last_active).total_seconds() > SESSION_EXPIRY_MINUTES * 60
        ]
        for sid in expired:
            del _sessions[sid]
        if expired:
            print(f"Cleaned up {len(expired)} expired sessions")


async def session_cleanup_task():
    """Background task to clean up expired sessions"""
    while True:
        await asyncio.sleep(60)  # Check every minute
        await cleanup_expired_sessions()


def generate_session_id() -> str:
    """Generate a cryptographically secure session ID"""
    return secrets.token_urlsafe(32)


async def get_or_create_session(session_id: Optional[str] = None) -> MCPSession:
    """Get existing session or create new one (single-tenant)"""
    async with _session_lock:
        if session_id and session_id in _sessions:
            session = _sessions[session_id]
            session.touch()
            return session

        # Create new session with shared handler
        new_session_id = generate_session_id()
        session = MCPSession(
            session_id=new_session_id,
            handler=mcp_handler,
        )
        _sessions[new_session_id] = session
        return session


async def get_session(session_id: str) -> Optional[MCPSession]:
    """Get session by ID"""
    async with _session_lock:
        session = _sessions.get(session_id)
        if session:
            session.touch()
        return session


async def delete_session(session_id: str) -> bool:
    """Delete a session"""
    async with _session_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            return True
        return False


# === Authentication Helper ===

async def verify_bearer_token(authorization: Optional[str]) -> bool:
    """
    Verify Bearer token for single-tenant mode.
    Returns True if valid or auth disabled.
    Raises HTTPException on failure.
    """
    if not auth_manager.auth_enabled:
        return True

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header format. Use: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1]
    is_valid, error = auth_manager.verify_token(token)
    if not is_valid:
        raise HTTPException(
            status_code=401,
            detail=error,
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


# === Lifecycle ===

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the app"""
    print("=" * 60)
    print("  SimpleMem MCP Server (Single-Tenant)")
    print("  S3-backed Memory Service for LLM Agents")
    print("=" * 60)
    print()
    print(f"  LLM Model: {settings.llm_model}")
    print(f"  Embedding Model: {settings.embedding_model}")
    print(f"  S3 Bucket: {settings.s3_bucket}")
    print(f"  Auth: {'Enabled' if auth_manager.auth_enabled else 'Disabled'}")
    print(f"  Transport: Streamable HTTP (2025-03-26)")
    print()

    # Test S3 connection
    s3_ok, s3_msg = vector_store.test_connection()
    print(f"  S3 Connection: {s3_msg}")
    print()
    print("-" * 60)

    # Start session cleanup task
    cleanup_task = asyncio.create_task(session_cleanup_task())

    yield

    # Cancel cleanup task
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # Close OpenRouter client
    await openrouter_client.close()

    print("SimpleMem MCP Server stopped")


# === FastAPI App ===

app = FastAPI(
    title="SimpleMem MCP Server",
    description="Single-Tenant Memory Service for LLM Agents with S3 Storage",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Health & Info ===

@app.get("/api/health")
async def health_check():
    """Health check endpoint with S3 connectivity test"""
    s3_ok, s3_msg = vector_store.test_connection()

    return {
        "status": "healthy" if s3_ok else "degraded",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "s3_storage": "connected" if s3_ok else f"error: {s3_msg}",
    }


@app.get("/api/server/info")
async def server_info():
    """Get server information"""
    return {
        "name": "SimpleMem MCP Server",
        "version": "1.0.0",
        "mode": "single-tenant",
        "protocol_version": "2025-03-26",
        "transport": "Streamable HTTP",
        "storage": "S3",
        "s3_bucket": settings.s3_bucket,
        "llm_model": settings.llm_model,
        "embedding_model": settings.embedding_model,
        "auth_enabled": auth_manager.auth_enabled,
        "active_sessions": len(_sessions),
    }


# === MCP Protocol Helper Functions ===

def _is_initialize_request(data: dict | list) -> bool:
    """Check if the message is an initialize request"""
    if isinstance(data, list):
        return any(
            isinstance(item, dict) and item.get("method") == "initialize"
            for item in data
        )
    return isinstance(data, dict) and data.get("method") == "initialize"


def _is_notification_or_response_only(data: dict | list) -> bool:
    """Check if message contains only notifications or responses (no requests)"""
    items = data if isinstance(data, list) else [data]
    for item in items:
        if not isinstance(item, dict):
            continue
        # Has 'method' but no 'id' -> notification
        # Has 'result' or 'error' -> response
        # Has both 'method' and 'id' -> request
        if "method" in item and "id" in item:
            return False  # This is a request
    return True


# === MCP Protocol Endpoints (Streamable HTTP - 2025-03-26 spec) ===

@app.post("/mcp")
async def mcp_post_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """
    Streamable HTTP POST endpoint (MCP 2025-03-26 spec).

    Handles JSON-RPC 2.0 messages from clients.

    Headers:
    - Authorization: Bearer <token> (required if SIMPLEMEM_ACCESS_KEY is set)
    - Accept: application/json, text/event-stream (required)
    - Mcp-Session-Id: <session-id> (required after initialization)

    Request body: JSON-RPC request, notification, response, or array of them.

    Response:
    - For notifications/responses only: 202 Accepted
    - For requests: JSON response or SSE stream
    """
    # Validate Accept header
    accept = request.headers.get("accept", "")
    if "application/json" not in accept and "text/event-stream" not in accept:
        raise HTTPException(
            status_code=406,
            detail="Accept header must include application/json or text/event-stream",
        )

    # Authenticate
    await verify_bearer_token(authorization)

    # Parse request body
    try:
        body = await request.body()
        data = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as e:
        return JSONResponse(
            status_code=400,
            content={
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {e}"},
            },
        )

    # Handle initialization (creates new session)
    if _is_initialize_request(data):
        session = await get_or_create_session()
        session.initialized = True

        # Process the initialize request
        response_str = await session.handler.handle_message(json.dumps(data))
        response_data = json.loads(response_str)

        # Return with Mcp-Session-Id header
        return JSONResponse(
            content=response_data,
            headers={"Mcp-Session-Id": session.session_id},
        )

    # For non-initialization requests, try to get existing session or create new one
    session = None
    if mcp_session_id:
        session = await get_session(mcp_session_id)

    # Auto-create session if not found (resilient mode)
    # This handles cases where session expired or server restarted
    session_recreated = False
    if not session:
        session = await get_or_create_session()
        session.initialized = True  # Mark as initialized since client already initialized
        session_recreated = True
        logger.warning(
            f"Session not found or expired, created new session {session.session_id} "
            f"(previous session_id: {mcp_session_id or 'none'})"
        )

    # If only notifications or responses, return 202 Accepted
    if _is_notification_or_response_only(data):
        # Still process them (e.g., initialized notification)
        await session.handler.handle_message(json.dumps(data))
        return Response(status_code=202)

    # Process request(s) and return response
    response_str = await session.handler.handle_message(json.dumps(data))
    response_data = json.loads(response_str)

    # Build response headers
    response_headers = {"Mcp-Session-Id": session.session_id}
    if session_recreated:
        response_headers["X-Session-Recreated"] = "true"

    return JSONResponse(
        content=response_data,
        headers=response_headers,
    )


@app.get("/mcp")
async def mcp_get_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
):
    """
    Streamable HTTP GET endpoint for server-to-client SSE stream.

    Used for server-initiated messages (notifications, requests to client).

    Headers:
    - Authorization: Bearer <token> (required if SIMPLEMEM_ACCESS_KEY is set)
    - Accept: text/event-stream (required)
    - Mcp-Session-Id: <session-id> (required)
    - Last-Event-ID: <event-id> (optional, for resumability)
    """
    # Validate Accept header
    accept = request.headers.get("accept", "")
    if "text/event-stream" not in accept:
        raise HTTPException(
            status_code=406,
            detail="Accept header must include text/event-stream",
        )

    # Authenticate
    await verify_bearer_token(authorization)

    # Session ID required for GET
    if not mcp_session_id:
        raise HTTPException(
            status_code=400,
            detail="Mcp-Session-Id header required",
        )

    # Get session
    session = await get_session(mcp_session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired",
        )

    # Generate unique stream ID
    stream_id = secrets.token_urlsafe(16)
    session.active_streams.add(stream_id)

    async def event_generator():
        """Generate SSE events for server-to-client messages"""
        try:
            # Send initial keepalive
            yield ": keepalive\n\n"

            while True:
                try:
                    # Wait for messages with timeout for keepalive
                    message = await asyncio.wait_for(
                        session.message_queue.get(),
                        timeout=15.0,
                    )

                    # Format as SSE event
                    event_id = session.next_event_id()
                    yield f"id: {event_id}\n"
                    yield f"event: message\n"
                    yield f"data: {json.dumps(message)}\n\n"

                except asyncio.TimeoutError:
                    # Send keepalive comment
                    yield ": keepalive\n\n"

        finally:
            # Remove stream from active set
            session.active_streams.discard(stream_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Mcp-Session-Id": session.session_id,
        },
    )


@app.delete("/mcp")
async def mcp_delete_endpoint(
    authorization: Optional[str] = Header(None),
    mcp_session_id: Optional[str] = Header(None, alias="Mcp-Session-Id"),
):
    """
    Terminate an MCP session.

    Headers:
    - Authorization: Bearer <token> (required if SIMPLEMEM_ACCESS_KEY is set)
    - Mcp-Session-Id: <session-id> (required)
    """
    # Authenticate
    await verify_bearer_token(authorization)

    if not mcp_session_id:
        raise HTTPException(status_code=400, detail="Mcp-Session-Id header required")

    # Get session to verify it exists
    session = await get_session(mcp_session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Delete session
    await delete_session(mcp_session_id)
    return Response(status_code=204)


# === Memory Viewer Dashboard ===

@app.get("/memories", response_class=HTMLResponse)
async def memory_viewer():
    """Memory Viewer Dashboard - View all stored memories"""
    s3_ok, _ = vector_store.test_connection()

    # Get all entries for display (this is the source of truth)
    try:
        entries = await vector_store.get_all_entries()
        # Use actual entries count, not cached stats
        entry_count = len(entries)

        # Collect unique values for filters (escaped for safe HTML output)
        all_topics = set()
        all_persons = set()
        all_entities = set()
        for e in entries:
            if e.topic:
                all_topics.add(html.escape(e.topic))
            for p in (e.persons or []):
                all_persons.add(html.escape(p))
            for ent in (e.entities or []):
                all_entities.add(html.escape(ent))

        entries_data = [
            {
                # Escape all user-provided content to prevent XSS attacks
                "entry_id": html.escape(e.entry_id) if e.entry_id else "",
                "content": html.escape(e.lossless_restatement) if e.lossless_restatement else "",
                "timestamp": html.escape(e.timestamp) if e.timestamp else "—",
                "created_at": html.escape(e.created_at) if e.created_at else "—",
                "location": html.escape(e.location) if e.location else "—",
                "persons": html.escape(", ".join(e.persons)) if e.persons else "—",
                "persons_list": [html.escape(p) for p in (e.persons or [])],
                "entities": html.escape(", ".join(e.entities)) if e.entities else "—",
                "entities_list": [html.escape(ent) for ent in (e.entities or [])],
                "topic": html.escape(e.topic) if e.topic else "—",
                "keywords": html.escape(", ".join(e.keywords)) if e.keywords else "—",
            }
            for e in entries
        ]
    except Exception as ex:
        entries_data = []
        entry_count = 0
        all_topics = set()
        all_persons = set()
        all_entities = set()
        print(f"Error loading entries: {ex}")

    # Build memory cards HTML
    memory_cards = ""
    for i, entry in enumerate(entries_data):
        # Convert lists to JSON for data attributes
        persons_json = json.dumps(entry["persons_list"])
        entities_json = json.dumps(entry["entities_list"])
        memory_cards += f'''
        <div class="memory-card" data-index="{i}" data-entry-id="{entry["entry_id"]}" data-topic="{entry["topic"]}" data-persons='{persons_json}' data-entities='{entities_json}'>
            <div class="card-header">
                <input type="checkbox" class="memory-checkbox" data-entry-id="{entry["entry_id"]}" onchange="updateSelectionCount()">
                <span class="card-index">#{i + 1}</span>
            </div>
            <div class="memory-content">{entry["content"]}</div>
            <div class="memory-meta">
                <div class="meta-row">
                    <span class="meta-label">Event Time:</span>
                    <span class="meta-value">{entry["timestamp"]}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-label">Stored:</span>
                    <span class="meta-value">{entry["created_at"]}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-label">Persons:</span>
                    <span class="meta-value tag-list">{entry["persons"]}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-label">Location:</span>
                    <span class="meta-value">{entry["location"]}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-label">Entities:</span>
                    <span class="meta-value">{entry["entities"]}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-label">Topic:</span>
                    <span class="meta-value topic">{entry["topic"]}</span>
                </div>
                <div class="meta-row">
                    <span class="meta-label">Keywords:</span>
                    <span class="meta-value tag-list">{entry["keywords"]}</span>
                </div>
            </div>
        </div>
        '''

    if not memory_cards:
        memory_cards = '''
        <div class="empty-state">
            <div class="empty-icon">🧠</div>
            <h3>No Memories Yet</h3>
            <p>Start storing memories using the MCP tools:</p>
            <code>memory_add(speaker="user", content="Your information here")</code>
        </div>
        '''

    html = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SimpleMem - Memory Viewer</title>
        <style>
            * {{ box-sizing: border-box; margin: 0; padding: 0; }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                min-height: 100vh;
                color: #e4e4e7;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                padding: 24px;
            }}
            header {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 32px;
                flex-wrap: wrap;
                gap: 16px;
            }}
            .logo {{
                display: flex;
                align-items: center;
                gap: 12px;
            }}
            .logo h1 {{
                font-size: 28px;
                font-weight: 700;
                background: linear-gradient(135deg, #60a5fa, #a78bfa);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .logo-icon {{
                font-size: 32px;
            }}
            nav {{
                display: flex;
                gap: 12px;
            }}
            nav a {{
                color: #a1a1aa;
                text-decoration: none;
                padding: 8px 16px;
                border-radius: 8px;
                transition: all 0.2s;
            }}
            nav a:hover, nav a.active {{
                color: #fff;
                background: rgba(255,255,255,0.1);
            }}

            /* Stats Bar */
            .stats-bar {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-bottom: 24px;
            }}
            .stat-card {{
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 36px;
                font-weight: 700;
                background: linear-gradient(135deg, #60a5fa, #34d399);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }}
            .stat-label {{
                color: #a1a1aa;
                font-size: 14px;
                margin-top: 4px;
            }}
            .stat-sublabel {{
                color: #52525b;
                font-size: 11px;
                margin-top: 2px;
                font-style: italic;
            }}
            .stat-card.status .stat-value {{
                font-size: 18px;
                background: none;
                -webkit-text-fill-color: {"#22c55e" if s3_ok else "#ef4444"};
            }}
            .stat-value.small {{
                font-size: 14px;
                word-break: break-all;
            }}

            /* Controls */
            .controls {{
                display: flex;
                gap: 12px;
                margin-bottom: 24px;
                flex-wrap: wrap;
            }}
            .search-box {{
                flex: 1;
                min-width: 250px;
                position: relative;
            }}
            .search-box input {{
                width: 100%;
                padding: 12px 16px 12px 44px;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 10px;
                background: rgba(255,255,255,0.05);
                color: #fff;
                font-size: 15px;
                transition: all 0.2s;
            }}
            .search-box input:focus {{
                outline: none;
                border-color: #60a5fa;
                background: rgba(255,255,255,0.08);
            }}
            .search-box::before {{
                content: "🔍";
                position: absolute;
                left: 14px;
                top: 50%;
                transform: translateY(-50%);
                font-size: 16px;
            }}
            .btn {{
                padding: 12px 20px;
                border: none;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.2s;
                display: flex;
                align-items: center;
                gap: 8px;
            }}
            .btn-primary {{
                background: linear-gradient(135deg, #3b82f6, #8b5cf6);
                color: white;
            }}
            .btn-primary:hover {{
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
            }}
            .btn-danger {{
                background: rgba(239, 68, 68, 0.2);
                color: #f87171;
                border: 1px solid rgba(239, 68, 68, 0.3);
            }}
            .btn-danger:hover {{
                background: rgba(239, 68, 68, 0.3);
            }}
            .btn-secondary {{
                background: rgba(255, 255, 255, 0.1);
                color: #a1a1aa;
                border: 1px solid rgba(255,255,255,0.2);
            }}
            .btn-secondary:hover {{
                background: rgba(255, 255, 255, 0.15);
                color: #e4e4e7;
            }}
            .btn:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
                transform: none !important;
            }}

            /* Filter Selects */
            .filter-select {{
                padding: 12px 16px;
                border: 1px solid rgba(255,255,255,0.2);
                border-radius: 10px;
                background: rgba(255,255,255,0.05);
                color: #fff;
                font-size: 14px;
                cursor: pointer;
                min-width: 140px;
            }}
            .filter-select:focus {{
                outline: none;
                border-color: #60a5fa;
            }}
            .filter-select option {{
                background: #1a1a2e;
                color: #fff;
            }}

            /* Selection Controls */
            .selection-controls {{
                display: flex;
                gap: 12px;
                margin-bottom: 16px;
                align-items: center;
                flex-wrap: wrap;
                padding: 12px 16px;
                background: rgba(255,255,255,0.03);
                border-radius: 10px;
                border: 1px solid rgba(255,255,255,0.08);
            }}
            .select-all-label {{
                display: flex;
                align-items: center;
                gap: 8px;
                color: #a1a1aa;
                cursor: pointer;
                font-size: 14px;
            }}
            .select-all-label:hover {{
                color: #e4e4e7;
            }}
            .selection-count {{
                color: #60a5fa;
                font-weight: 600;
                font-size: 14px;
                margin-left: auto;
            }}

            /* Card Header */
            .card-header {{
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 12px;
            }}
            .memory-checkbox {{
                width: 18px;
                height: 18px;
                cursor: pointer;
                accent-color: #60a5fa;
            }}
            .card-index {{
                font-size: 12px;
                color: #71717a;
                font-weight: 600;
            }}
            .memory-card.selected {{
                border-color: #60a5fa;
                background: rgba(96, 165, 250, 0.1);
            }}

            /* Memory Grid */
            .memory-grid {{
                display: grid;
                gap: 16px;
            }}
            .memory-card {{
                background: rgba(255,255,255,0.03);
                border: 1px solid rgba(255,255,255,0.08);
                border-radius: 16px;
                padding: 24px;
                transition: all 0.3s;
            }}
            .memory-card:hover {{
                background: rgba(255,255,255,0.06);
                border-color: rgba(96, 165, 250, 0.3);
                transform: translateY(-2px);
            }}
            .memory-content {{
                font-size: 16px;
                line-height: 1.6;
                color: #f4f4f5;
                margin-bottom: 16px;
                padding-bottom: 16px;
                border-bottom: 1px solid rgba(255,255,255,0.08);
                word-break: break-word;
                overflow-wrap: break-word;
            }}
            .memory-meta {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 12px;
            }}
            .meta-row {{
                display: flex;
                flex-direction: column;
                gap: 4px;
            }}
            .meta-label {{
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                color: #71717a;
            }}
            .meta-value {{
                font-size: 14px;
                color: #a1a1aa;
                word-break: break-word;
                overflow-wrap: break-word;
            }}
            .meta-value.tag-list {{
                color: #60a5fa;
            }}
            .meta-value.topic {{
                color: #a78bfa;
            }}

            /* Empty State */
            .empty-state {{
                text-align: center;
                padding: 60px 20px;
                background: rgba(255,255,255,0.02);
                border: 2px dashed rgba(255,255,255,0.1);
                border-radius: 16px;
            }}
            .empty-icon {{
                font-size: 64px;
                margin-bottom: 16px;
            }}
            .empty-state h3 {{
                font-size: 24px;
                margin-bottom: 8px;
                color: #e4e4e7;
            }}
            .empty-state p {{
                color: #71717a;
                margin-bottom: 16px;
            }}
            .empty-state code {{
                display: inline-block;
                background: rgba(96, 165, 250, 0.1);
                color: #60a5fa;
                padding: 8px 16px;
                border-radius: 8px;
                font-size: 13px;
            }}

            /* Footer */
            footer {{
                margin-top: 48px;
                padding-top: 24px;
                border-top: 1px solid rgba(255,255,255,0.08);
                text-align: center;
                color: #52525b;
                font-size: 13px;
            }}
            footer a {{
                color: #60a5fa;
                text-decoration: none;
            }}

            /* Modal */
            .modal-overlay {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.7);
                z-index: 1000;
                align-items: center;
                justify-content: center;
            }}
            .modal-overlay.active {{
                display: flex;
            }}
            .modal {{
                background: #1e1e2e;
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 16px;
                padding: 32px;
                max-width: 400px;
                text-align: center;
            }}
            .modal h3 {{
                font-size: 20px;
                margin-bottom: 12px;
            }}
            .modal p {{
                color: #a1a1aa;
                margin-bottom: 24px;
            }}
            .modal-buttons {{
                display: flex;
                gap: 12px;
                justify-content: center;
            }}

            /* Responsive */
            @media (max-width: 640px) {{
                .container {{ padding: 16px; }}
                header {{ flex-direction: column; align-items: flex-start; }}
                .memory-meta {{ grid-template-columns: 1fr; }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <div class="logo">
                    <span class="logo-icon">🧠</span>
                    <h1>SimpleMem</h1>
                </div>
                <nav>
                    <a href="/">Status</a>
                    <a href="/memories" class="active">Memories</a>
                    <a href="/api/health">API Health</a>
                </nav>
            </header>

            <div class="stats-bar">
                <div class="stat-card">
                    <div class="stat-value">{entry_count}</div>
                    <div class="stat-label">Total Memories</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{settings.s3_bucket}</div>
                    <div class="stat-label">S3 Bucket</div>
                </div>
                <div class="stat-card status">
                    <div class="stat-value">{"● Connected" if s3_ok else "● Disconnected"}</div>
                    <div class="stat-label">Storage Status</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value small">{settings.llm_model}</div>
                    <div class="stat-label">LLM Model</div>
                    <div class="stat-sublabel">Memory extraction & answers</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value small">{settings.embedding_model}</div>
                    <div class="stat-label">Embedding Model</div>
                    <div class="stat-sublabel">Vector indexing & search</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{settings.embedding_dimension}</div>
                    <div class="stat-label">Vector Dimension</div>
                </div>
            </div>

            <div class="controls">
                <div class="search-box">
                    <input type="text" id="searchInput" placeholder="Search memories..." onkeyup="filterMemories()">
                </div>
                <select id="topicFilter" class="filter-select" onchange="filterMemories()">
                    <option value="">All Topics</option>
                    {''.join(f'<option value="{t}">{t}</option>' for t in sorted(all_topics))}
                </select>
                <select id="personFilter" class="filter-select" onchange="filterMemories()">
                    <option value="">All Persons</option>
                    {''.join(f'<option value="{p}">{p}</option>' for p in sorted(all_persons))}
                </select>
                <select id="entityFilter" class="filter-select" onchange="filterMemories()">
                    <option value="">All Entities</option>
                    {''.join(f'<option value="{e}">{e}</option>' for e in sorted(all_entities))}
                </select>
                <button class="btn btn-secondary" onclick="clearFilters()">
                    Clear Filters
                </button>
            </div>

            <div class="selection-controls">
                <label class="select-all-label">
                    <input type="checkbox" id="selectAll" onchange="toggleSelectAll()">
                    Select All Visible
                </label>
                <span id="selectionCount" class="selection-count">0 selected</span>
                <button class="btn btn-danger" id="deleteSelectedBtn" onclick="showDeleteSelectedModal()" disabled>
                    🗑 Delete Selected
                </button>
                <button class="btn btn-primary" onclick="location.reload()">
                    ↻ Refresh
                </button>
                <button class="btn btn-danger" onclick="showClearModal()">
                    🗑 Clear All
                </button>
            </div>

            <div class="memory-grid" id="memoryGrid">
                {memory_cards}
            </div>

            <footer>
                <p>SimpleMem v1.0.0 · <a href="https://aiming-lab.github.io/SimpleMem-Page/" target="_blank">Documentation</a></p>
            </footer>
        </div>

        <!-- Clear Confirmation Modal -->
        <div class="modal-overlay" id="clearModal">
            <div class="modal">
                <h3>⚠️ Clear All Memories?</h3>
                <p>This action cannot be undone. All {entry_count} memories will be permanently deleted.</p>
                <div class="modal-buttons">
                    <button class="btn btn-primary" onclick="hideClearModal()">Cancel</button>
                    <button class="btn btn-danger" onclick="clearAllMemories()">Delete All</button>
                </div>
            </div>
        </div>

        <!-- Delete Selected Confirmation Modal -->
        <div class="modal-overlay" id="deleteSelectedModal">
            <div class="modal">
                <h3>⚠️ Delete Selected Memories?</h3>
                <p id="deleteSelectedCount">This action cannot be undone.</p>
                <div class="modal-buttons">
                    <button class="btn btn-primary" onclick="hideDeleteSelectedModal()">Cancel</button>
                    <button class="btn btn-danger" onclick="deleteSelectedMemories()">Delete Selected</button>
                </div>
            </div>
        </div>

        <script>
            // Filter memories by text search and dropdown filters
            function filterMemories() {{
                const query = document.getElementById('searchInput').value.toLowerCase();
                const topicFilter = document.getElementById('topicFilter').value;
                const personFilter = document.getElementById('personFilter').value;
                const entityFilter = document.getElementById('entityFilter').value;
                const cards = document.querySelectorAll('.memory-card');

                cards.forEach(card => {{
                    const content = card.textContent.toLowerCase();
                    const topic = card.dataset.topic || '';
                    const persons = JSON.parse(card.dataset.persons || '[]');
                    const entities = JSON.parse(card.dataset.entities || '[]');

                    // Check text search
                    const matchesText = !query || content.includes(query);

                    // Check topic filter
                    const matchesTopic = !topicFilter || topic === topicFilter;

                    // Check person filter
                    const matchesPerson = !personFilter || persons.includes(personFilter);

                    // Check entity filter
                    const matchesEntity = !entityFilter || entities.includes(entityFilter);

                    // Show if all filters match
                    card.style.display = (matchesText && matchesTopic && matchesPerson && matchesEntity) ? 'block' : 'none';
                }});

                // Update select all checkbox state
                updateSelectAllState();
            }}

            function clearFilters() {{
                document.getElementById('searchInput').value = '';
                document.getElementById('topicFilter').value = '';
                document.getElementById('personFilter').value = '';
                document.getElementById('entityFilter').value = '';
                filterMemories();
            }}

            // Selection functions
            function toggleSelectAll() {{
                const selectAll = document.getElementById('selectAll').checked;
                const visibleCards = document.querySelectorAll('.memory-card[style="display: block"], .memory-card:not([style*="display"])');

                visibleCards.forEach(card => {{
                    if (card.style.display !== 'none') {{
                        const checkbox = card.querySelector('.memory-checkbox');
                        if (checkbox) {{
                            checkbox.checked = selectAll;
                            card.classList.toggle('selected', selectAll);
                        }}
                    }}
                }});

                updateSelectionCount();
            }}

            function updateSelectAllState() {{
                const visibleCheckboxes = [];
                document.querySelectorAll('.memory-card').forEach(card => {{
                    if (card.style.display !== 'none') {{
                        const checkbox = card.querySelector('.memory-checkbox');
                        if (checkbox) visibleCheckboxes.push(checkbox);
                    }}
                }});

                const allChecked = visibleCheckboxes.length > 0 && visibleCheckboxes.every(cb => cb.checked);
                document.getElementById('selectAll').checked = allChecked;
            }}

            function updateSelectionCount() {{
                const selected = document.querySelectorAll('.memory-checkbox:checked');
                const count = selected.length;
                document.getElementById('selectionCount').textContent = count + ' selected';

                const deleteBtn = document.getElementById('deleteSelectedBtn');
                deleteBtn.disabled = count === 0;

                // Update card styling
                document.querySelectorAll('.memory-card').forEach(card => {{
                    const checkbox = card.querySelector('.memory-checkbox');
                    card.classList.toggle('selected', checkbox && checkbox.checked);
                }});

                updateSelectAllState();
            }}

            // Modal functions
            function showClearModal() {{
                document.getElementById('clearModal').classList.add('active');
            }}

            function hideClearModal() {{
                document.getElementById('clearModal').classList.remove('active');
            }}

            function showDeleteSelectedModal() {{
                const count = document.querySelectorAll('.memory-checkbox:checked').length;
                document.getElementById('deleteSelectedCount').textContent =
                    'This action cannot be undone. ' + count + ' memor' + (count === 1 ? 'y' : 'ies') + ' will be permanently deleted.';
                document.getElementById('deleteSelectedModal').classList.add('active');
            }}

            function hideDeleteSelectedModal() {{
                document.getElementById('deleteSelectedModal').classList.remove('active');
            }}

            async function clearAllMemories() {{
                try {{
                    // Initialize session first
                    const initRes = await fetch('/mcp', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Accept': 'application/json'
                        }},
                        body: JSON.stringify({{
                            jsonrpc: '2.0',
                            method: 'initialize',
                            id: 1,
                            params: {{}}
                        }})
                    }});
                    const sessionId = initRes.headers.get('Mcp-Session-Id');

                    // Call clear tool
                    await fetch('/mcp', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'Accept': 'application/json',
                            'Mcp-Session-Id': sessionId
                        }},
                        body: JSON.stringify({{
                            jsonrpc: '2.0',
                            method: 'tools/call',
                            id: 2,
                            params: {{
                                name: 'memory_clear',
                                arguments: {{}}
                            }}
                        }})
                    }});

                    location.reload();
                }} catch (err) {{
                    alert('Error clearing memories: ' + err.message);
                }}
            }}

            async function deleteSelectedMemories() {{
                try {{
                    const selectedCheckboxes = document.querySelectorAll('.memory-checkbox:checked');
                    const entryIds = Array.from(selectedCheckboxes).map(cb => cb.dataset.entryId);

                    if (entryIds.length === 0) {{
                        hideDeleteSelectedModal();
                        return;
                    }}

                    const response = await fetch('/api/memories/delete', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{ entry_ids: entryIds }})
                    }});

                    const result = await response.json();

                    if (result.success) {{
                        location.reload();
                    }} else {{
                        alert('Error deleting memories: ' + (result.error || 'Unknown error'));
                        hideDeleteSelectedModal();
                    }}
                }} catch (err) {{
                    alert('Error deleting memories: ' + err.message);
                    hideDeleteSelectedModal();
                }}
            }}

            // Close modals on escape key
            document.addEventListener('keydown', (e) => {{
                if (e.key === 'Escape') {{
                    hideClearModal();
                    hideDeleteSelectedModal();
                }}
            }});
        </script>
    </body>
    </html>
    '''
    return HTMLResponse(content=html)


# === API Endpoint for Memory Data (JSON) ===

@app.get("/api/memories")
async def get_memories_api():
    """Get all memories as JSON for programmatic access"""
    try:
        entries = await vector_store.get_all_entries()
        return {
            "success": True,
            "count": len(entries),
            "memories": [
                {
                    "content": e.lossless_restatement,
                    "timestamp": e.timestamp,
                    "location": e.location,
                    "persons": e.persons,
                    "entities": e.entities,
                    "topic": e.topic,
                }
                for e in entries
            ]
        }
    except Exception as ex:
        return {"success": False, "error": str(ex), "memories": []}


@app.post("/api/memories/delete")
async def delete_memories_api(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Delete specific memories by their entry IDs"""
    # Authenticate (consistent with /mcp endpoints)
    await verify_bearer_token(authorization)

    try:
        body = await request.json()
        entry_ids = body.get("entry_ids", [])

        if not entry_ids:
            return {"success": False, "error": "No entry_ids provided", "deleted": 0}

        deleted_count = await vector_store.delete_entries(entry_ids)
        return {"success": True, "deleted": deleted_count}
    except Exception as ex:
        return {"success": False, "error": str(ex), "deleted": 0}


# === Root Endpoint (Single-Tenant Status Page) ===

@app.get("/", response_class=HTMLResponse)
async def root_status():
    """Show server status page (single-tenant mode)"""
    s3_ok, _ = vector_store.test_connection()
    status_color = "#22c55e" if s3_ok else "#ef4444"

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SimpleMem MCP Server</title>
        <style>
            body {{ font-family: system-ui, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }}
            h1 {{ color: #1f2937; }}
            .status {{ display: inline-block; padding: 4px 12px; border-radius: 4px; background: {status_color}; color: white; }}
            .info {{ background: #f3f4f6; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            code {{ background: #e5e7eb; padding: 2px 6px; border-radius: 4px; }}
        </style>
    </head>
    <body>
        <h1>SimpleMem MCP Server</h1>
        <p>Efficient Lifelong Memory for LLM Agents</p>
        <p>Status: <span class="status">{"Running" if s3_ok else "Degraded"}</span></p>

        <div class="info">
            <p><strong>Mode:</strong> Single-Tenant</p>
            <p><strong>Storage:</strong> S3 ({settings.s3_bucket})</p>
            <p><strong>Auth:</strong> {"Enabled" if auth_manager.auth_enabled else "Disabled"}</p>
            <p><strong>MCP Endpoint:</strong> <code>POST /mcp</code></p>
        </div>

        <h3>Quick Test</h3>
        <pre style="background: #1f2937; color: #f3f4f6; padding: 15px; border-radius: 8px; overflow-x: auto;">
curl -X POST {settings.mcp_base_url or 'http://localhost:' + str(settings.port)}/mcp \\
  -H "Content-Type: application/json" \\
  -H "Accept: application/json" \\
  -d '{{"jsonrpc":"2.0","method":"initialize","id":1,"params":{{}}}}'</pre>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# === Entry Point ===

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the HTTP server"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()


# =============================================================================
# MULTI-TENANT HTTP SERVER CODE (PRESERVED FOR REFERENCE)
# See multi-tenant-backup branch for original implementation
# =============================================================================
#
# Key differences in multi-tenant mode:
# - User registration endpoint (/api/auth/register)
# - JWT token generation and verification
# - Per-user table isolation in LanceDB
# - UserStore for SQLite user metadata
# - OpenRouterClientManager for per-user API keys
# - Session management with user_id validation
#
# Removed endpoints:
# - POST /api/auth/register - User registration with OpenRouter API key
# - GET /api/auth/verify - Token verification
# - POST /api/auth/refresh - Token refresh
# - GET /mcp/sse - Legacy SSE endpoint
# - POST /mcp/message - Legacy message endpoint
# =============================================================================
