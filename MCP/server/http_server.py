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
import json
import os
import secrets
from typing import Optional
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Request, Header, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse

from .auth.token_manager import SimpleAuthManager
from .database.vector_store import SingleTenantVectorStore
from .integrations.openrouter import OpenRouterClient
from .mcp_handler import MCPHandler

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_settings


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

    # For non-initialization requests, session ID is required
    if not mcp_session_id:
        raise HTTPException(
            status_code=400,
            detail="Mcp-Session-Id header required for non-initialization requests",
        )

    # Get existing session
    session = await get_session(mcp_session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Send new InitializeRequest.",
        )

    # If only notifications or responses, return 202 Accepted
    if _is_notification_or_response_only(data):
        # Still process them (e.g., initialized notification)
        await session.handler.handle_message(json.dumps(data))
        return Response(status_code=202)

    # Process request(s) and return response
    response_str = await session.handler.handle_message(json.dumps(data))
    response_data = json.loads(response_str)

    return JSONResponse(
        content=response_data,
        headers={"Mcp-Session-Id": session.session_id},
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


# === Static Files (Frontend) ===

frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve frontend HTML"""
    index_path = os.path.join(frontend_path, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>SimpleMem MCP Server</h1><p>Single-tenant mode with S3 storage.</p>")


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
