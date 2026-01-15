"""
Settings configuration for SimpleMem MCP Server - Single Tenant Mode

This configuration supports S3-compatible storage (DigitalOcean Spaces)
for LanceDB vector database persistence.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache


@dataclass
class Settings:
    """Application settings for single-tenant deployment"""

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")

    # Single-Tenant Authentication
    # Optional access key for client authentication (if not set, no auth required)
    simplemem_access_key: Optional[str] = field(
        default_factory=lambda: os.getenv("SIMPLEMEM_ACCESS_KEY")
    )

    # OpenRouter Configuration (server-side only)
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL", "openai/gpt-4.1-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "qwen/qwen3-embedding-4b")
    )
    embedding_dimension: int = 2560

    # S3 Storage Configuration (DigitalOcean Spaces)
    s3_bucket: str = field(default_factory=lambda: os.getenv("S3_BUCKET", ""))
    s3_access_key: str = field(default_factory=lambda: os.getenv("S3_ACCESS_KEY", ""))
    s3_secret_key: str = field(default_factory=lambda: os.getenv("S3_SECRET_KEY", ""))
    s3_endpoint: str = field(
        default_factory=lambda: os.getenv("S3_ENDPOINT", "https://nyc3.digitaloceanspaces.com")
    )
    s3_region: str = field(default_factory=lambda: os.getenv("S3_REGION", "nyc3"))

    # LanceDB path (computed from S3 settings in __post_init__)
    lancedb_path: str = ""

    # Single-tenant table name (fixed)
    table_name: str = "memories"

    # Memory Building Configuration
    window_size: int = 20
    overlap_size: int = 2

    # Retrieval Configuration
    semantic_top_k: int = 25
    keyword_top_k: int = 5
    enable_planning: bool = True
    enable_reflection: bool = True
    max_reflection_rounds: int = 2

    # LLM Configuration
    llm_temperature: float = 0.1
    llm_max_retries: int = 3
    use_streaming: bool = True

    def __post_init__(self):
        """Validate required configuration and compute derived values"""
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is required. "
                "Get one at https://openrouter.ai/keys"
            )

        if not self.s3_bucket:
            raise ValueError(
                "S3_BUCKET environment variable is required. "
                "Create a DigitalOcean Spaces bucket first."
            )

        if not self.s3_access_key or not self.s3_secret_key:
            raise ValueError(
                "S3_ACCESS_KEY and S3_SECRET_KEY environment variables are required. "
                "Generate Spaces keys in DigitalOcean Control Panel -> API -> Spaces Keys."
            )

        # Construct LanceDB S3 path
        self.lancedb_path = f"s3://{self.s3_bucket}/lancedb"

    def get_s3_storage_options(self) -> dict:
        """Get storage options for LanceDB S3 connection"""
        return {
            "aws_access_key_id": self.s3_access_key,
            "aws_secret_access_key": self.s3_secret_key,
            "aws_endpoint": self.s3_endpoint,
            "aws_region": self.s3_region,
        }


# Clear the lru_cache when needed (for testing)
def clear_settings_cache():
    """Clear the settings cache"""
    get_settings.cache_clear()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# =============================================================================
# MULTI-TENANT CONFIGURATION (PRESERVED FOR REFERENCE)
# See multi-tenant-backup branch for original implementation
# =============================================================================
#
# The original multi-tenant configuration included:
#
# # JWT Configuration
# jwt_secret_key: str = field(default_factory=lambda: os.getenv(
#     "JWT_SECRET_KEY",
#     "simplemem-secret-key-change-in-production"
# ))
# jwt_algorithm: str = "HS256"
# jwt_expiration_days: int = 30
#
# # Encryption for API Keys
# encryption_key: str = field(default_factory=lambda: os.getenv(
#     "ENCRYPTION_KEY",
#     "simplemem-encryption-key-32bytes!"  # Must be 32 bytes for AES-256
# ))
#
# # Database Paths (local)
# data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "./data"))
# lancedb_path: str = field(default_factory=lambda: os.getenv("LANCEDB_PATH", "./data/lancedb"))
# user_db_path: str = field(default_factory=lambda: os.getenv("USER_DB_PATH", "./data/users.db"))
#
# def __post_init__(self):
#     """Ensure directories exist"""
#     os.makedirs(self.data_dir, exist_ok=True)
#     os.makedirs(self.lancedb_path, exist_ok=True)
# =============================================================================
