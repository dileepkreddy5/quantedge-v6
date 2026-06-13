"""
QuantEdge v6.0 — Local Authentication (replaces AWS Cognito)
=============================================================
Single-user, self-hosted JWT auth. No external identity provider.

Auth lives behind an `AuthProvider` interface. `LocalJWTProvider` is the
current implementation (single owner, bcrypt-hashed password, HS256 JWTs
signed with SECRET_KEY). Swapping to multi-user / a managed IdP later means
writing one new provider class — routers never change.

Re-exports the same public names the old cognito_auth.py exposed so the 12
routers that import them need no edits.
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Dict

import bcrypt
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from pydantic import BaseModel
from loguru import logger

from core.config import settings


security = HTTPBearer(auto_error=True)

_ALGO = "HS256"
_ACCESS_TTL = settings.ACCESS_TOKEN_EXPIRE_HOURS * 3600
_REFRESH_TTL = settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400


class CognitoUser(BaseModel):
    """Kept this name for drop-in compatibility. It's just 'the user'."""
    username: str
    email: str
    sub: str


class AuthProvider(ABC):
    @abstractmethod
    def login(self, username: str, password: str) -> Dict: ...
    @abstractmethod
    def verify_token(self, token: str) -> Dict: ...
    @abstractmethod
    def refresh_tokens(self, refresh_token: str) -> Dict: ...


def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def _check_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except Exception:
        return False


class LocalJWTProvider(AuthProvider):
    """
    Single-owner JWT auth. Credentials from settings:
      OWNER_USERNAME, OWNER_EMAIL, OWNER_PASSWORD_HASH (bcrypt hash).
    If OWNER_PASSWORD_HASH is unset but OWNER_PASSWORD is set, hash at boot.
    """

    def __init__(self):
        self.username = settings.OWNER_USERNAME
        self.email = settings.OWNER_EMAIL
        self.sub = "owner-" + settings.OWNER_USERNAME

        pw_hash = getattr(settings, "OWNER_PASSWORD_HASH", None)
        pw_plain = getattr(settings, "OWNER_PASSWORD", None)
        if pw_hash:
            self._pw_hash = pw_hash
        elif pw_plain:
            logger.warning(
                "OWNER_PASSWORD_HASH not set — hashing OWNER_PASSWORD at boot. "
                "For production set OWNER_PASSWORD_HASH in .env."
            )
            self._pw_hash = _hash_password(pw_plain)
        else:
            logger.error("No OWNER_PASSWORD_HASH or OWNER_PASSWORD set — login will reject all.")
            self._pw_hash = ""

    def _make_token(self, kind: str, ttl: int) -> str:
        now = int(time.time())
        payload = {
            "sub": self.sub, "username": self.username, "email": self.email,
            "token_use": kind, "iat": now, "exp": now + ttl,
            "jti": uuid.uuid4().hex, "iss": "quantedge-local",
        }
        return jwt.encode(payload, settings.SECRET_KEY, algorithm=_ALGO)

    def login(self, username: str, password: str) -> Dict:
        if username.lower() != self.username.lower() or not self._pw_hash:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        if not _check_password(password, self._pw_hash):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
        return {
            "access_token": self._make_token("access", _ACCESS_TTL),
            "refresh_token": self._make_token("refresh", _REFRESH_TTL),
            "expires_in": _ACCESS_TTL, "token_type": "Bearer",
        }

    def verify_token(self, token: str) -> Dict:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[_ALGO], issuer="quantedge-local")
        except JWTError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail=f"Token validation failed: {e}",
                                headers={"WWW-Authenticate": "Bearer"})
        if payload.get("username", "").lower() != self.username.lower():
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access restricted to authorized user")
        return payload

    def refresh_tokens(self, refresh_token: str) -> Dict:
        payload = self.verify_token(refresh_token)
        if payload.get("token_use") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not a refresh token")
        return {"access_token": self._make_token("access", _ACCESS_TTL), "expires_in": _ACCESS_TTL}


class CognitoAuthenticator:
    """Shim preserving the old interface; delegates to the active AuthProvider."""
    def __init__(self):
        self.provider: AuthProvider = LocalJWTProvider()

    def login(self, username: str, password: str) -> Dict:
        return self.provider.login(username, password)

    async def verify_token(self, token: str) -> Dict:
        return self.provider.verify_token(token)

    def refresh_tokens(self, refresh_token: str) -> Dict:
        return self.provider.refresh_tokens(refresh_token)

    def respond_to_mfa(self, session: str, username: str, mfa_code: str) -> Dict:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="MFA is not enabled in local mode")


_auth = CognitoAuthenticator()


def get_cognito_auth() -> CognitoAuthenticator:
    return _auth


def verify_cognito_token(token: str) -> Dict:
    return _auth.provider.verify_token(token)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CognitoUser:
    token = credentials.credentials
    redis = request.app.state.redis
    if await redis.get(f"revoked:{token[:32]}"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Session has been revoked. Please login again.")
    payload = await _auth.verify_token(token)
    return CognitoUser(username=payload.get("username", ""),
                       email=payload.get("email", ""), sub=payload.get("sub", ""))


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = None,
) -> Optional[CognitoUser]:
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.replace("Bearer ", "").strip()
    if not token:
        return None
    try:
        claims = await _auth.verify_token(token)
        return CognitoUser(username=claims.get("username", ""),
                           email=claims.get("email", ""), sub=claims.get("sub", ""))
    except Exception:
        return None
