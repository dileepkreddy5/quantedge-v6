"""QuantEdge v5.0 — Portfolio Router"""
from fastapi import APIRouter, Depends
from auth.cognito_auth import get_current_user, CognitoUser

router = APIRouter()

@router.get("/portfolio")
async def get_portfolio(current_user: CognitoUser = Depends(get_current_user)):
    return {"portfolio": [], "message": "Connect your broker to enable portfolio tracking"}
