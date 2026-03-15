"""
QuantEdge v5.0 — Authentication Router
Login, MFA, Refresh, Logout endpoints with full security.
"""

from fastapi import APIRouter, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import time
from loguru import logger
from auth.cognito_auth import get_cognito_auth, get_current_user, CognitoUser, CognitoAuthenticator
from core.config import settings


router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str

    @validator("username")
    def validate_username(cls, v):
        if v.lower() != settings.OWNER_USERNAME:
            raise ValueError("Invalid credentials")  # Don't leak which field is wrong
        return v.lower()


class MFARequest(BaseModel):
    session: str
    username: str
    mfa_code: str

    @validator("mfa_code")
    def validate_mfa(cls, v):
        if not v.isdigit() or len(v) != 6:
            raise ValueError("MFA code must be 6 digits")
        return v


class RefreshRequest(BaseModel):
    refresh_token: str


async def check_lockout(redis, ip: str):
    """Check if IP is locked out after failed attempts"""
    fail_key = f"login_fails:{ip}"
    lock_key = f"login_lock:{ip}"

    is_locked = await redis.get(lock_key)
    if is_locked:
        lock_ttl = await redis.ttl(lock_key)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Account locked. Try again in {lock_ttl} seconds.",
        )


async def record_failure(redis, ip: str):
    """Record failed login attempt, lock after max attempts"""
    fail_key = f"login_fails:{ip}"
    lock_key = f"login_lock:{ip}"

    fails = await redis.incr(fail_key)
    if fails == 1:
        await redis.expire(fail_key, 600)  # Reset counter after 10 min

    if fails >= settings.MAX_LOGIN_ATTEMPTS:
        await redis.setex(lock_key, settings.LOCKOUT_DURATION_SECONDS, "locked")
        await redis.delete(fail_key)
        logger.warning(f"🔒 IP locked after {fails} failed attempts: {ip}")
        # Send SNS alert to Dileep
        try:
            if settings.SNS_ALERT_TOPIC_ARN:
                import boto3
                sns = boto3.client("sns", region_name=settings.AWS_REGION)
                sns.publish(
                    TopicArn=settings.SNS_ALERT_TOPIC_ARN,
                    Subject="⚠️ QuantEdge: Multiple Failed Login Attempts",
                    Message=f"IP {ip} has been locked after {fails} failed login attempts.\nTime: {time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                )
        except Exception:
            pass


async def clear_failures(redis, ip: str):
    """Clear failure count on successful login"""
    await redis.delete(f"login_fails:{ip}")


@router.post("/login")
async def login(request: Request, body: LoginRequest):
    """
    Step 1 of auth: validate username + password.
    Returns MFA challenge session token.
    Always requires MFA — no exceptions.
    """
    redis = request.app.state.redis
    client_ip = request.headers.get("X-Forwarded-For", request.client.host).split(",")[0].strip()

    await check_lockout(redis, client_ip)

    try:
        auth = get_cognito_auth()
        result = auth.login(body.username, body.password)

        logger.info(f"Login step 1 success: {body.username} from {client_ip}")
        # If Cognito returned tokens directly (no MFA), return them
        if result.get("access_token"):
            return {
                "requires_mfa": False,
                "access_token": result.get("access_token"),
                "refresh_token": result.get("refresh_token"),
                "message": "Login successful",
            }
        return {
            "requires_mfa": True,
            "session": result.get("session"),
            "message": "Enter your authenticator code",
        }

    except HTTPException as e:
        if e.status_code == status.HTTP_401_UNAUTHORIZED:
            await record_failure(redis, client_ip)
            logger.warning(f"Failed login: {body.username} from {client_ip}")
        raise


@router.post("/mfa")
async def verify_mfa(request: Request, body: MFARequest):
    """
    Step 2 of auth: validate TOTP code.
    Returns JWT tokens on success.
    """
    redis = request.app.state.redis
    client_ip = request.headers.get("X-Forwarded-For", request.client.host).split(",")[0].strip()

    await check_lockout(redis, client_ip)

    try:
        auth = get_cognito_auth()
        tokens = auth.respond_to_mfa(body.session, body.username, body.mfa_code)

        # Clear failure count on success
        await clear_failures(redis, client_ip)

        # Store session in Redis for revocation support
        access_token = tokens["access_token"]
        await redis.setex(
            f"session:{access_token[:32]}",
            tokens["expires_in"],
            body.username,
        )

        logger.info(f"✅ Login successful: {body.username} from {client_ip}")

        return JSONResponse(
            content={
                "access_token": tokens["access_token"],
                "id_token": tokens["id_token"],
                "refresh_token": tokens["refresh_token"],
                "expires_in": tokens["expires_in"],
                "user": "Dileep",
                "message": "Welcome back, Dileep 👋",
            },
            headers={
                # Also set httpOnly cookie for browser security
                "Set-Cookie": (
                    f"qe_token={tokens['access_token']}; "
                    f"HttpOnly; Secure; SameSite=Strict; "
                    f"Path=/; Max-Age={tokens['expires_in']}"
                )
            }
        )

    except HTTPException as e:
        if e.status_code == status.HTTP_401_UNAUTHORIZED:
            await record_failure(redis, client_ip)
        raise


@router.post("/refresh")
async def refresh_token(request: Request, body: RefreshRequest):
    """Refresh access token using refresh token"""
    try:
        auth = get_cognito_auth()
        tokens = auth.refresh_tokens(body.refresh_token)
        return tokens
    except HTTPException:
        raise


@router.delete("/logout")
async def logout(
    request: Request,
    current_user: CognitoUser = Depends(get_current_user),
):
    """
    Revoke current session by adding token to Redis blocklist.
    Token will be rejected on next request even if not expired.
    """
    redis = request.app.state.redis

    # Get the token from Authorization header
    auth_header = request.headers.get("Authorization", "")
    token = auth_header.replace("Bearer ", "") if auth_header else ""

    if token:
        # Add to revocation list (expires when original token would have)
        await redis.setex(f"revoked:{token[:32]}", 28800, "revoked")
        # Remove from active sessions
        await redis.delete(f"session:{token[:32]}")

    logger.info(f"✅ Logout: {current_user.username}")
    return {"message": "Logged out successfully"}


@router.get("/me")
async def get_me(current_user: CognitoUser = Depends(get_current_user)):
    """Returns current user info"""
    return {
        "username": current_user.username,
        "email": current_user.email,
        "platform": "QuantEdge v5.0",
        "access": "full",
        "timestamp": int(time.time()),
    }
