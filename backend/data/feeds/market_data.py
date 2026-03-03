"""
QuantEdge v5.0 — Market Data Feed
====================================
Fetches real OHLCV, fundamentals, options data.
Primary: yFinance (free, real-time)
Fallback: Alpha Vantage (25 calls/day free tier)
Premium: Polygon.io (when available)
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import Dict, Optional
import time
from loguru import logger
from core.config import settings

try:
    import yfinance as yf
    YFINANCE_OK = True
except ImportError:
    YFINANCE_OK = False


class MarketDataFeed:
    """Primary market data source using yFinance"""

    async def get_price_history(
        self,
        ticker: str,
        years: int = 10,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        Returns DataFrame with columns: open, high, low, close, volume
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_sync, ticker, years, interval)

    def _fetch_sync(self, ticker: str, years: int, interval: str) -> pd.DataFrame:
        if not YFINANCE_OK:
            return pd.DataFrame()
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=f"{years}y", interval=interval)
            if df.empty:
                return pd.DataFrame()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna()
            df.index = pd.to_datetime(df.index)
            df = df[df["close"] > 0]
            logger.info(f"✅ Price data: {ticker} — {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
            return df
        except Exception as e:
            logger.error(f"Price fetch error {ticker}: {e}")
            return pd.DataFrame()

    async def get_realtime_quote(self, ticker: str) -> Dict:
        """Current price, volume, bid/ask"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_quote_sync, ticker)

    def _fetch_quote_sync(self, ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fast = stock.fast_info
            return {
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "open": info.get("open"),
                "high": info.get("dayHigh"),
                "low": info.get("dayLow"),
                "volume": info.get("volume"),
                "avg_volume": info.get("averageVolume"),
                "market_cap": info.get("marketCap"),
                "bid": info.get("bid"),
                "ask": info.get("ask"),
                "bid_size": info.get("bidSize"),
                "ask_size": info.get("askSize"),
                "pre_market": info.get("preMarketPrice"),
                "after_hours": info.get("postMarketPrice"),
            }
        except Exception as e:
            logger.error(f"Quote fetch error {ticker}: {e}")
            return {}


class FundamentalDataFeed:
    """Fundamental data from yFinance"""

    async def get_fundamentals(self, ticker: str) -> Dict:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_fundamentals_sync, ticker)

    def _fetch_fundamentals_sync(self, ticker: str) -> Dict:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Financials
            try:
                income = stock.income_stmt
                balance = stock.balance_sheet
                cashflow = stock.cashflow
            except Exception:
                income = balance = cashflow = None

            def safe_get(d, key, default=None):
                try:
                    v = d.get(key, default)
                    return float(v) if v is not None else default
                except Exception:
                    return default

            fundamentals = {
                # Identity
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "exchange": info.get("exchange"),
                "country": info.get("country"),
                "description": (info.get("longBusinessSummary", "") or "")[:500],

                # Valuation
                "pe_ratio": safe_get(info, "trailingPE"),
                "forward_pe": safe_get(info, "forwardPE"),
                "peg_ratio": safe_get(info, "pegRatio"),
                "price_to_book": safe_get(info, "priceToBook"),
                "price_to_sales": safe_get(info, "priceToSalesTrailing12Months"),
                "ev_ebitda": safe_get(info, "enterpriseToEbitda"),
                "ev_revenue": safe_get(info, "enterpriseToRevenue"),
                "enterprise_value": safe_get(info, "enterpriseValue"),
                "market_cap": safe_get(info, "marketCap"),

                # Trading
                "week_52_high": safe_get(info, "fiftyTwoWeekHigh"),
                "week_52_low": safe_get(info, "fiftyTwoWeekLow"),
                "beta": safe_get(info, "beta"),
                "short_interest": safe_get(info, "shortPercentOfFloat"),
                "shares_outstanding": safe_get(info, "sharesOutstanding"),
                "float_shares": safe_get(info, "floatShares"),
                "shares_short": safe_get(info, "sharesShort"),
                "institutional_ownership": safe_get(info, "institutionPercentHeld"),
                "insider_ownership": safe_get(info, "insiderPercentHeld"),

                # Profitability
                "gross_margin": safe_get(info, "grossMargins"),
                "operating_margin": safe_get(info, "operatingMargins"),
                "ebitda_margin": safe_get(info, "ebitdaMargins"),
                "net_margin": safe_get(info, "profitMargins"),
                "roe": safe_get(info, "returnOnEquity"),
                "roa": safe_get(info, "returnOnAssets"),
                "roic": None,  # Calculated below

                # Growth
                "revenue_growth": safe_get(info, "revenueGrowth"),
                "earnings_growth": safe_get(info, "earningsGrowth"),
                "revenue_ttm": safe_get(info, "totalRevenue"),
                "ebitda": safe_get(info, "ebitda"),
                "free_cash_flow": safe_get(info, "freeCashflow"),
                "eps_ttm": safe_get(info, "trailingEps"),
                "eps_forward": safe_get(info, "forwardEps"),

                # Balance Sheet
                "total_debt": safe_get(info, "totalDebt"),
                "total_cash": safe_get(info, "totalCash"),
                "debt_to_equity": safe_get(info, "debtToEquity"),
                "current_ratio": safe_get(info, "currentRatio"),
                "quick_ratio": safe_get(info, "quickRatio"),
                "interest_coverage": None,

                # Dividends
                "dividend_yield": safe_get(info, "dividendYield"),
                "dividend_rate": safe_get(info, "dividendRate"),
                "payout_ratio": safe_get(info, "payoutRatio"),

                # Earnings
                "earnings_date": str(info.get("earningsTimestamp", "")),
                "earnings_quarterly_growth": safe_get(info, "earningsQuarterlyGrowth"),
                "surprise_history": None,  # From EDGAR pipeline
            }

            # Calculate ROIC if possible
            if fundamentals["ebitda"] and fundamentals["total_debt"] is not None:
                invested_capital = (fundamentals["total_debt"] or 0) + (safe_get(info, "totalStockholderEquity") or 0)
                if invested_capital > 0:
                    nopat = (fundamentals["ebitda"] or 0) * 0.7
                    fundamentals["roic"] = nopat / invested_capital

            # FCF yield
            if fundamentals["free_cash_flow"] and fundamentals["market_cap"]:
                fundamentals["fcf_yield"] = fundamentals["free_cash_flow"] / fundamentals["market_cap"]

            # FCF margin
            if fundamentals["free_cash_flow"] and fundamentals["revenue_ttm"]:
                fundamentals["fcf_margin"] = fundamentals["free_cash_flow"] / fundamentals["revenue_ttm"]

            return {k: v for k, v in fundamentals.items() if v is not None}

        except Exception as e:
            logger.error(f"Fundamentals fetch error {ticker}: {e}")
            return {"name": ticker}


class OptionsDataFeed:
    """Real options chain data from yFinance"""

    async def get_chain(self, ticker: str, max_expiries: int = 6) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_chain_sync, ticker, max_expiries)

    def _fetch_chain_sync(self, ticker: str, max_expiries: int) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            if not expirations:
                return pd.DataFrame()

            all_rows = []
            for exp in expirations[:max_expiries]:
                try:
                    chain = stock.option_chain(exp)
                    dte = (pd.Timestamp(exp) - pd.Timestamp.now()).days

                    # Process calls
                    calls = chain.calls.copy()
                    calls["option_type"] = "call"
                    calls["expiry"] = exp
                    calls["days_to_expiry"] = dte

                    # Process puts
                    puts = chain.puts.copy()
                    puts["option_type"] = "put"
                    puts["expiry"] = exp
                    puts["days_to_expiry"] = dte

                    # Merge calls/puts by strike for GEX calculation
                    for _, call_row in calls.iterrows():
                        k = call_row.get("strike", 0)
                        put_row = puts[puts["strike"] == k]
                        row = {
                            "strike": k,
                            "days_to_expiry": dte,
                            "iv": float(call_row.get("impliedVolatility", 0.25)),
                            "open_interest_call": float(call_row.get("openInterest", 0)),
                            "volume_call": float(call_row.get("volume", 0)),
                            "open_interest_put": float(put_row["openInterest"].iloc[0]) if len(put_row) > 0 else 0,
                            "volume_put": float(put_row["volume"].iloc[0]) if len(put_row) > 0 else 0,
                        }
                        all_rows.append(row)
                except Exception:
                    continue

            if not all_rows:
                return pd.DataFrame()

            df = pd.DataFrame(all_rows)
            df = df[df["days_to_expiry"] > 0]
            logger.info(f"✅ Options: {ticker} — {len(df)} strikes across {max_expiries} expiries")
            return df

        except Exception as e:
            logger.error(f"Options fetch error {ticker}: {e}")
            return pd.DataFrame()


class SentimentDataFeed:
    """News and Reddit sentiment data"""

    async def get_news_and_reddit(self, ticker: str) -> Dict:
        tasks = await asyncio.gather(
            self._get_news(ticker),
            self._get_reddit(ticker),
            return_exceptions=True,
        )
        news = tasks[0] if not isinstance(tasks[0], Exception) else []
        reddit = tasks[1] if not isinstance(tasks[1], Exception) else []
        return {"news": news, "reddit": reddit}

    async def _get_news(self, ticker: str) -> list:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_news_sync, ticker)

    def _fetch_news_sync(self, ticker: str) -> list:
        try:
            stock = yf.Ticker(ticker)
            news = stock.news or []
            return [{
                "title": n.get("title", ""),
                "publisher": n.get("publisher", ""),
                "timestamp": n.get("providerPublishTime", 0),
                "url": n.get("link", ""),
            } for n in news[:10]]
        except Exception:
            return []

    async def _get_reddit(self, ticker: str) -> list:
        """Fetch recent Reddit posts mentioning the ticker"""
        if not settings.REDDIT_CLIENT_ID or not settings.REDDIT_CLIENT_SECRET:
            return []
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._fetch_reddit_sync, ticker)
        except Exception:
            return []

    def _fetch_reddit_sync(self, ticker: str) -> list:
        try:
            import praw
            reddit = praw.Reddit(
                client_id=settings.REDDIT_CLIENT_ID,
                client_secret=settings.REDDIT_CLIENT_SECRET,
                user_agent="QuantEdge/5.0 (by Dileep)",
            )
            posts = []
            for sub in ["wallstreetbets", "investing", "stocks"]:
                subreddit = reddit.subreddit(sub)
                for post in subreddit.search(ticker, limit=10, time_filter="week"):
                    posts.append({
                        "title": post.title,
                        "body": (post.selftext or "")[:500],
                        "score": post.score,
                        "num_comments": post.num_comments,
                        "subreddit": sub,
                        "timestamp": post.created_utc,
                    })
            return posts[:30]
        except Exception as e:
            logger.debug(f"Reddit fetch error: {e}")
            return []
