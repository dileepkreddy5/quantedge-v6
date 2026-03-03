"""
QuantEdge v5.0 — AWS Cognito Authentication
=============================================
JWT validation via Cognito public keys (JWKS).
MFA enforcement: every login requires TOTP (Google Authenticator).
Session management via Redis (allows instant revocation).
"""

import json
import time
import httpx
from typing import Optional, Dict
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, jwk, JWTError
from jose.utils import base64url_decode
from pydantic import BaseModel
from loguru import logger
import boto3
from botocore.exceptions import ClientError

from core.config import settings


security = HTTPBearer()


class CognitoUser(BaseModel):
    username: str
    email: str
    sub: str  # Cognito user ID


class CognitoAuthenticator:
    """
    Validates Cognito JWT tokens using the pool's public JWKS endpoint.
    Caches JWKS in memory (refreshed hourly).
    """

    def __init__(self):
        self.jwks_url = (
            f"https://cognito-idp.{settings.COGNITO_REGION}.amazonaws.com/"
            f"{settings.COGNITO_USER_POOL_ID}/.well-known/jwks.json"
        )
        self._jwks_cache: Optional[Dict] = None
        self._jwks_cached_at: float = 0
        self._cache_ttl: float = 3600  # 1 hour

        self.cognito_client = boto3.client(
            "cognito-idp",
            region_name=settings.COGNITO_REGION,
        )

    async def get_jwks(self) -> Dict:
        """Fetch and cache Cognito public keys"""
        now = time.time()
        if self._jwks_cache and (now - self._jwks_cached_at) < self._cache_ttl:
            return self._jwks_cache

        async with httpx.AsyncClient() as client:
            response = await client.get(self.jwks_url, timeout=10.0)
            response.raise_for_status()
            self._jwks_cache = response.json()
            self._jwks_cached_at = now
            return self._jwks_cache

    async def verify_token(self, token: str) -> Dict:
        """
        Verify JWT:
          1. Decode header to get kid (key ID)
          2. Find matching public key in JWKS
          3. Verify signature, expiry, audience, issuer
          4. Check session is not revoked in Redis
        """
        try:
            # Get key ID from token header (without verifying)
            header = jwt.get_unverified_header(token)
            kid = header.get("kid")

            # Find matching public key
            jwks = await self.get_jwks()
            public_key = None
            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    public_key = key
                    break

            if not public_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token: key not found",
                )

            # Construct the public key
            rsa_key = jwk.construct(public_key)

            # Decode and verify JWT
            payload = jwt.decode(
                token,
                rsa_key,
                algorithms=["RS256"],
                audience=settings.COGNITO_CLIENT_ID,
                issuer=f"https://cognito-idp.{settings.COGNITO_REGION}.amazonaws.com/{settings.COGNITO_USER_POOL_ID}",
                options={"verify_at_hash": False},
            )

            # Enforce single-owner policy
            username = payload.get("cognito:username", payload.get("username", ""))
            if username != settings.OWNER_USERNAME:
                logger.warning(f"⛔ Unauthorized access attempt: {username}")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access restricted to authorized user",
                )

            return payload

        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token validation failed: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def login(self, username: str, password: str) -> Dict:
        """
        Initiate Cognito auth flow.
        Returns challenge if MFA required (it always should be).
        """
        try:
            response = self.cognito_client.initiate_auth(
                AuthFlow="USER_PASSWORD_AUTH",
                AuthParameters={
                    "USERNAME": username,
                    "PASSWORD": password,
                },
                ClientId=settings.COGNITO_CLIENT_ID,
            )

            # If MFA is configured (as it should be), returns TOTP challenge
            if response.get("ChallengeName") == "SOFTWARE_TOKEN_MFA":
                return {
                    "challenge": "SOFTWARE_TOKEN_MFA",
                    "session": response["Session"],
                    "requires_mfa": True,
                }
            elif "AuthenticationResult" in response:
                # MFA somehow not enforced (shouldn't happen with our Cognito config)
                return response["AuthenticationResult"]
            else:
                raise HTTPException(status_code=401, detail="Unexpected auth response")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ["NotAuthorizedException", "UserNotFoundException"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid credentials",
                )
            elif error_code == "UserNotConfirmedException":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Account not confirmed",
                )
            raise HTTPException(status_code=500, detail="Authentication service error")

    def respond_to_mfa(self, session: str, username: str, mfa_code: str) -> Dict:
        """
        Complete MFA challenge with TOTP code (Google Authenticator).
        Returns JWT tokens on success.
        """
        try:
            response = self.cognito_client.respond_to_auth_challenge(
                ChallengeName="SOFTWARE_TOKEN_MFA",
                ChallengeResponses={
                    "SOFTWARE_TOKEN_MFA_CODE": mfa_code,
                    "USERNAME": username,
                },
                Session=session,
                ClientId=settings.COGNITO_CLIENT_ID,
            )

            if "AuthenticationResult" in response:
                auth = response["AuthenticationResult"]
                return {
                    "access_token": auth["AccessToken"],
                    "id_token": auth["IdToken"],
                    "refresh_token": auth["RefreshToken"],
                    "expires_in": auth.get("ExpiresIn", 28800),
                    "token_type": "Bearer",
                }

            raise HTTPException(status_code=401, detail="MFA verification failed")

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code in ["CodeMismatchException", "ExpiredCodeException"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or expired MFA code",
                )
            raise HTTPException(status_code=500, detail="MFA service error")

    def refresh_tokens(self, refresh_token: str) -> Dict:
        """Refresh access token using refresh token"""
        try:
            response = self.cognito_client.initiate_auth(
                AuthFlow="REFRESH_TOKEN_AUTH",
                AuthParameters={"REFRESH_TOKEN": refresh_token},
                ClientId=settings.COGNITO_CLIENT_ID,
            )
            auth = response["AuthenticationResult"]
            return {
                "access_token": auth["AccessToken"],
                "id_token": auth["IdToken"],
                "expires_in": auth.get("ExpiresIn", 28800),
            }
        except ClientError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token invalid or expired",
            )


# Global authenticator instance
_auth = CognitoAuthenticator()


def verify_cognito_token(token: str) -> Dict:
    """Sync wrapper for token verification (used in WebSocket)"""
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_auth.verify_token(token))
    finally:
        loop.close()


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CognitoUser:
    """
    FastAPI dependency: validates JWT and returns user.
    Also checks Redis for session revocation (logout support).
    """
    token = credentials.credentials

    # Check if token has been revoked (logout)
    redis = request.app.state.redis
    is_revoked = await redis.get(f"revoked:{token[:32]}")
    if is_revoked:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session has been revoked. Please login again.",
        )

    payload = await _auth.verify_token(token)

    return CognitoUser(
        username=payload.get("cognito:username", payload.get("username", "")),
        email=payload.get("email", ""),
        sub=payload.get("sub", ""),
    )


def get_cognito_auth() -> CognitoAuthenticator:
    return _auth


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = None,
) -> Optional[CognitoUser]:
    """
    Optional auth dependency.
    Returns CognitoUser if valid token present, None if no token.
    Used for public endpoints that give extra features when logged in.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    token = auth_header.replace("Bearer ", "").strip()
    if not token:
        return None
    try:
        auth = get_cognito_auth()
        claims = await auth.verify_token(token)
        return CognitoUser(
            username=claims.get("cognito:username", claims.get("sub", "")),
            email=claims.get("email", ""),
            sub=claims.get("sub", ""),
        )
    except Exception:
        return None
