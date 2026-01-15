"""
Simple authentication manager for single-tenant mode.

Validates bearer tokens against the SIMPLEMEM_ACCESS_KEY environment variable.
If no access key is set, authentication is disabled and all requests are allowed.
"""

from typing import Optional, Tuple


class SimpleAuthManager:
    """
    Simple bearer token authentication for single-tenant mode.
    Validates against a configured access key.
    """

    def __init__(self, access_key: Optional[str] = None):
        """
        Initialize the auth manager.

        Args:
            access_key: The access key to validate against.
                       If None or empty, authentication is disabled.
        """
        self.access_key = access_key
        self.auth_enabled = access_key is not None and len(access_key) > 0

    def verify_token(self, token: str) -> Tuple[bool, Optional[str]]:
        """
        Verify bearer token against configured access key.

        Args:
            token: The bearer token to verify

        Returns:
            Tuple of (is_valid, error_message)
            - (True, None) if valid or auth disabled
            - (False, error_message) if invalid
        """
        if not self.auth_enabled:
            return True, None  # Auth disabled, allow all

        if not token:
            return False, "Missing access token"

        if token == self.access_key:
            return True, None

        return False, "Invalid access token"


# =============================================================================
# MULTI-TENANT TOKEN MANAGER (PRESERVED FOR REFERENCE)
# See multi-tenant-backup branch for original implementation
# =============================================================================
#
# import jwt
# import base64
# import hashlib
# from datetime import datetime, timedelta
# from cryptography.fernet import Fernet
#
# from .models import User, TokenPayload
#
#
# class TokenManager:
#     """Manages JWT tokens and API key encryption"""
#
#     def __init__(
#         self,
#         secret_key: str,
#         encryption_key: str,
#         algorithm: str = "HS256",
#         expiration_days: int = 30,
#     ):
#         self.secret_key = secret_key
#         self.algorithm = algorithm
#         self.expiration_days = expiration_days
#
#         # Create Fernet cipher for API key encryption
#         # Derive a valid Fernet key from the encryption key
#         key = hashlib.sha256(encryption_key.encode()).digest()
#         self.fernet = Fernet(base64.urlsafe_b64encode(key))
#
#     def encrypt_api_key(self, api_key: str) -> str:
#         """Encrypt an API key for storage"""
#         return self.fernet.encrypt(api_key.encode()).decode()
#
#     def decrypt_api_key(self, encrypted_key: str) -> str:
#         """Decrypt a stored API key"""
#         return self.fernet.decrypt(encrypted_key.encode()).decode()
#
#     def generate_token(self, user: User) -> str:
#         """Generate a JWT token for a user"""
#         expiration = datetime.utcnow() + timedelta(days=self.expiration_days)
#
#         payload = TokenPayload(
#             user_id=user.user_id,
#             table_name=user.table_name,
#             created_at=user.created_at.isoformat(),
#             exp=int(expiration.timestamp()),
#         )
#
#         return jwt.encode(
#             payload.to_dict(),
#             self.secret_key,
#             algorithm=self.algorithm,
#         )
#
#     def verify_token(self, token: str) -> Tuple[bool, Optional[TokenPayload], Optional[str]]:
#         """
#         Verify a JWT token
#
#         Returns:
#             Tuple of (is_valid, payload, error_message)
#         """
#         try:
#             decoded = jwt.decode(
#                 token,
#                 self.secret_key,
#                 algorithms=[self.algorithm],
#             )
#             payload = TokenPayload.from_dict(decoded)
#             return True, payload, None
#
#         except jwt.ExpiredSignatureError:
#             return False, None, "Token has expired"
#         except jwt.InvalidTokenError as e:
#             return False, None, f"Invalid token: {str(e)}"
#
#     def refresh_token(self, token: str) -> Tuple[Optional[str], Optional[str]]:
#         """
#         Refresh a valid token with new expiration
#
#         Returns:
#             Tuple of (new_token, error_message)
#         """
#         is_valid, payload, error = self.verify_token(token)
#         if not is_valid:
#             return None, error
#
#         # Create new token with same user info but new expiration
#         expiration = datetime.utcnow() + timedelta(days=self.expiration_days)
#
#         new_payload = TokenPayload(
#             user_id=payload.user_id,
#             table_name=payload.table_name,
#             created_at=payload.created_at,
#             exp=int(expiration.timestamp()),
#         )
#
#         new_token = jwt.encode(
#             new_payload.to_dict(),
#             self.secret_key,
#             algorithm=self.algorithm,
#         )
#
#         return new_token, None
# =============================================================================
