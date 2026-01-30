#!/usr/bin/env python3
"""
SimpleMem MCP Server - Single Tenant Mode

Provides:
- MCP over Streamable HTTP (/mcp)
- Health and info endpoints (/api/*)
- S3-backed LanceDB storage
"""

import argparse
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="SimpleMem MCP Server - Single-Tenant Memory Service"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind to (default: 8000 or PORT env var)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    args = parser.parse_args()

    # Validate required environment variables
    required_vars = ["OPENROUTER_API_KEY", "S3_BUCKET", "S3_ACCESS_KEY", "S3_SECRET_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print("=" * 60)
        print("  ERROR: Missing required environment variables")
        print("=" * 60)
        print()
        for var in missing:
            print(f"  - {var}")
        print()
        print("  See .env.example for configuration template.")
        print()
        print("  Quick start:")
        print("    1. Copy .env.example to .env")
        print("    2. Fill in your credentials")
        print("    3. Run again")
        print()
        return 1

    # Show startup info
    print("=" * 60)
    print("  SimpleMem MCP Server (Single-Tenant)")
    print("  S3-backed Memory Service for LLM Agents")
    print("=" * 60)
    print()
    print(f"  MCP Endpoint: http://localhost:{args.port}/mcp")
    print(f"  Health Check: http://localhost:{args.port}/api/health")
    print(f"  Server Info:  http://localhost:{args.port}/api/server/info")
    print()
    print(f"  S3 Bucket: {os.getenv('S3_BUCKET')}")
    print(f"  Auth: {'Enabled' if os.getenv('SIMPLEMEM_ACCESS_KEY') else 'Disabled'}")
    print()
    print("-" * 60)

    import uvicorn
    uvicorn.run(
        "server.http_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
