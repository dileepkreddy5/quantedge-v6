"""
QuantEdge Price Oracle
Real ML computations + Claude synthesis for stock price prediction.
"""
from .engine import PriceOracleEngine
from .router import router

__all__ = ["PriceOracleEngine", "router"]
