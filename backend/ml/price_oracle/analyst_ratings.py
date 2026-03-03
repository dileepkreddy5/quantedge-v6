"""
QuantEdge — Wall Street Analyst Ratings Module
═══════════════════════════════════════════════════════════════════════════
Fetches REAL analyst recommendations from yFinance, which aggregates
ratings from Goldman Sachs, Morgan Stanley, JPMorgan, UBS, Citi, etc.

Data available via yFinance:
  ticker.recommendations         → full history of individual firm ratings
  ticker.recommendations_summary → current month aggregates (strongBuy/buy/hold/sell)
  ticker.analyst_price_targets   → low/mean/high/current 12-month price targets

How Wall Street ratings work:
  - Each firm (GS, MS, JPM etc.) publishes a rating: Strong Buy, Buy, 
    Overweight, Outperform, Hold, Neutral, Underweight, Sell, Strong Sell
  - They update ratings when fundamentals change or earnings miss/beat
  - Upgrades → price often jumps; Downgrades → price often drops
  - Conflicts of interest: firms with banking relationships rate more bullish
    (Michaely & Womack 1999 — documented analyst bias)
  - Best signal: CHANGES in rating, not the rating itself
    A move from Hold → Buy is more bullish than a static Buy rating

Academic context:
  - Jegadeesh et al. (2004): analyst recommendations have predictive power
    but diminish quickly — front-run within 2-5 days of publication
  - The combined (consensus) rating is more reliable than any single analyst
  - Upgrades from Hold→Buy generate +3-4% abnormal returns on average
    Downgrades generate -4-5% abnormal returns
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


# Rating normalization — Wall Street uses many different labels
# Mapped to a standard 5-level scale
RATING_MAP = {
    # Strong Buy equivalents
    "strong buy":     5, "strong_buy":  5, "strongbuy":    5,
    "conviction buy": 5, "top pick":    5, "speculative buy": 5,
    
    # Buy equivalents  
    "buy":            4, "outperform":  4, "overweight":   4,
    "accumulate":     4, "add":         4, "positive":     4,
    "market outperform": 4,
    
    # Hold equivalents
    "hold":           3, "neutral":     3, "equal-weight": 3,
    "equalweight":    3, "market perform": 3, "sector perform": 3,
    "in-line":        3, "inline":      3, "peer perform":  3,
    "fair value":     3, "market weight": 3,
    
    # Sell equivalents
    "sell":           2, "underperform":2, "underweight":  2,
    "reduce":         2, "negative":    2, "below average": 2,
    
    # Strong Sell equivalents
    "strong sell":    1, "strong_sell": 1, "conviction sell": 1,
}

SCORE_LABELS = {
    5: ("STRONG BUY",  "#00c896"),
    4: ("BUY",         "#40dda0"),
    3: ("HOLD",        "#e8b84b"),
    2: ("SELL",        "#ff8090"),
    1: ("STRONG SELL", "#ff4060"),
}

# Well-known broker names for display
KNOWN_BROKERS = {
    "Goldman Sachs", "Morgan Stanley", "JPMorgan", "JP Morgan",
    "Citigroup", "Citi", "Bank of America", "BofA", "Merrill Lynch",
    "UBS", "Deutsche Bank", "Barclays", "Credit Suisse", "Wells Fargo",
    "Jefferies", "Raymond James", "Stifel", "Piper Sandler",
    "Needham", "Cantor Fitzgerald", "Cowen", "TD Cowen",
    "RBC Capital", "RBC", "HSBC", "Bernstein", "Evercore",
    "Guggenheim", "KeyBanc", "Mizuho", "Truist", "Oppenheimer",
}


class AnalystRatingsEngine:
    """
    Fetches and processes Wall Street analyst ratings.
    Returns structured data ready for display and Claude synthesis.
    """
    
    def fetch(self, ticker: str) -> Dict[str, Any]:
        """
        Main method. Fetches all analyst data for ticker.
        Returns structured dict with ratings, price targets, consensus, trend.
        """
        import yfinance as yf
        stk = yf.Ticker(ticker.upper())
        
        # Fetch all three data sources
        rec_summary = self._fetch_summary(stk)
        rec_history = self._fetch_history(stk)
        price_targets = self._fetch_price_targets(stk, stk.info or {})
        
        # Compute derived metrics
        consensus    = self._compute_consensus(rec_summary, rec_history)
        recent_moves = self._get_recent_moves(rec_history)
        momentum     = self._compute_rating_momentum(rec_history)
        
        return {
            "ticker":        ticker.upper(),
            "summary":       rec_summary,
            "recent_ratings": rec_history[:15],   # last 15 individual ratings
            "price_targets": price_targets,
            "consensus":     consensus,
            "recent_moves":  recent_moves,
            "momentum":      momentum,
            "wall_street_verdict": self._verdict(consensus, momentum, price_targets),
        }
    
    def _fetch_summary(self, stk) -> Dict:
        """Fetch current month aggregate: strongBuy/buy/hold/sell/strongSell counts."""
        try:
            raw = stk.recommendations_summary
            if raw is None or (hasattr(raw, 'empty') and raw.empty):
                return {}
            
            # Take most recent period (0m = current month)
            if isinstance(raw, pd.DataFrame):
                rec_list = raw.to_dict("records")
            else:
                rec_list = raw if isinstance(raw, list) else []
            
            current = next((r for r in rec_list if str(r.get("period","")) == "0m"), None)
            if not current:
                current = rec_list[0] if rec_list else {}
            
            return {
                "strong_buy":  int(current.get("strongBuy",  0)),
                "buy":         int(current.get("buy",         0)),
                "hold":        int(current.get("hold",        0)),
                "sell":        int(current.get("sell",        0)),
                "strong_sell": int(current.get("strongSell",  0)),
                "period":      current.get("period", "0m"),
                # Previous month for trend comparison
                "prev_month":  self._get_prev_month(rec_list),
            }
        except Exception:
            return {}
    
    def _get_prev_month(self, rec_list: list) -> Dict:
        """Get -1m data for trend comparison."""
        prev = next((r for r in rec_list if str(r.get("period","")) == "-1m"), None)
        if not prev:
            return {}
        return {
            "strong_buy":  int(prev.get("strongBuy", 0)),
            "buy":         int(prev.get("buy",        0)),
            "hold":        int(prev.get("hold",       0)),
            "sell":        int(prev.get("sell",       0)),
            "strong_sell": int(prev.get("strongSell", 0)),
        }
    
    def _fetch_history(self, stk) -> List[Dict]:
        """
        Fetch individual firm ratings history.
        Returns list of {firm, date, rating, score, action} sorted newest first.
        """
        try:
            raw = stk.recommendations
            if raw is None or (hasattr(raw, 'empty') and raw.empty):
                return []
            
            if isinstance(raw, pd.DataFrame):
                rows = raw.reset_index()
                results = []
                for _, row in rows.iterrows():
                    date_val  = row.get("Date", row.get("Datetime", None))
                    firm      = str(row.get("Firm", row.get("firm", "")))
                    to_grade  = str(row.get("To Grade", row.get("toGrade", row.get("Action", "")))).lower().strip()
                    from_grade= str(row.get("From Grade", row.get("fromGrade", ""))).lower().strip()
                    action    = str(row.get("Action", "")).lower()
                    
                    score = RATING_MAP.get(to_grade, 3)
                    label, color = SCORE_LABELS.get(score, ("HOLD", "#e8b84b"))
                    
                    # Determine action type
                    if action in ("init", "initiated"):
                        action_label = "INITIATED"
                        action_color = "#4da8ff"
                    elif action in ("up", "upgrade"):
                        action_label = "UPGRADED ↑"
                        action_color = "#00c896"
                    elif action in ("down", "downgrade"):
                        action_label = "DOWNGRADED ↓"
                        action_color = "#ff4060"
                    elif action in ("main", "maintained", "reiterated"):
                        action_label = "MAINTAINED"
                        action_color = "#e8b84b"
                    else:
                        action_label = action.upper() or "UPDATED"
                        action_color = "#5a5a72"
                    
                    # Format date
                    if date_val is not None:
                        try:
                            dt = pd.to_datetime(date_val)
                            date_str = dt.strftime("%b %d, %Y")
                            days_ago = (datetime.now() - dt.to_pydatetime().replace(tzinfo=None)).days
                        except Exception:
                            date_str, days_ago = str(date_val)[:10], 999
                    else:
                        date_str, days_ago = "Unknown", 999
                    
                    results.append({
                        "firm":         firm,
                        "date":         date_str,
                        "days_ago":     days_ago,
                        "rating":       label,
                        "rating_raw":   to_grade,
                        "from_rating":  from_grade.title() if from_grade else None,
                        "score":        score,
                        "color":        color,
                        "action":       action_label,
                        "action_color": action_color,
                        "is_recent":    days_ago <= 30,
                        "is_known_firm": any(k.lower() in firm.lower() for k in KNOWN_BROKERS),
                    })
                
                # Sort newest first
                results.sort(key=lambda x: x["days_ago"])
                return results
        except Exception:
            return []
        return []
    
    def _fetch_price_targets(self, stk, info: dict) -> Dict:
        """Fetch analyst price targets (12-month)."""
        try:
            # Method 1: analyst_price_targets (newer yFinance versions)
            apt = getattr(stk, "analyst_price_targets", None)
            if apt is not None and not (hasattr(apt, 'empty') and apt.empty):
                if isinstance(apt, pd.DataFrame):
                    apt = apt.to_dict("records")[0] if len(apt) > 0 else {}
                return {
                    "low":     float(apt.get("low",    0) or 0),
                    "mean":    float(apt.get("mean",   0) or 0),
                    "high":    float(apt.get("high",   0) or 0),
                    "current": float(apt.get("current",0) or 0),
                    "median":  float(apt.get("median", 0) or 0),
                    "source":  "yfinance_apt",
                }
        except Exception:
            pass
        
        # Method 2: From info dict
        try:
            return {
                "low":     float(info.get("targetLowPrice",    0) or 0),
                "mean":    float(info.get("targetMeanPrice",   0) or 0),
                "high":    float(info.get("targetHighPrice",   0) or 0),
                "current": float(info.get("currentPrice",      0) or 0),
                "median":  float(info.get("targetMedianPrice", 0) or 0),
                "source":  "yfinance_info",
            }
        except Exception:
            return {}
    
    def _compute_consensus(self, summary: Dict, history: List[Dict]) -> Dict:
        """
        Compute weighted consensus score.
        - Summary counts: straightforward weighted average
        - Recency-weighted: recent ratings count more (30-day half-life)
        """
        # From summary counts
        sb = summary.get("strong_buy",  0)
        b  = summary.get("buy",         0)
        h  = summary.get("hold",        0)
        s  = summary.get("sell",        0)
        ss = summary.get("strong_sell", 0)
        total = sb + b + h + s + ss
        
        if total > 0:
            raw_score = (5*sb + 4*b + 3*h + 2*s + 1*ss) / total
            buy_pct   = (sb + b) / total * 100
            sell_pct  = (ss + s) / total * 100
            hold_pct  = h / total * 100
        else:
            raw_score = 3.0
            buy_pct = sell_pct = hold_pct = 33.3
        
        # Recency-weighted from history (weight decays with age)
        if history:
            weights, scores = [], []
            for r in history[:30]:  # last 30 ratings
                age   = max(r["days_ago"], 1)
                w     = np.exp(-age / 30)   # 30-day half-life
                weights.append(w)
                scores.append(r["score"])
            
            recency_score = float(np.average(scores, weights=weights)) if weights else raw_score
        else:
            recency_score = raw_score
        
        # Blend: 60% recency-weighted, 40% count-weighted
        final_score = 0.6 * recency_score + 0.4 * raw_score
        
        # Map score to label
        if final_score >= 4.2:   label, color = "STRONG BUY",  "#00c896"
        elif final_score >= 3.6: label, color = "BUY",         "#40dda0"
        elif final_score >= 2.6: label, color = "HOLD",        "#e8b84b"
        elif final_score >= 1.8: label, color = "SELL",        "#ff8090"
        else:                    label, color = "STRONG SELL", "#ff4060"
        
        return {
            "score":         round(final_score, 2),
            "label":         label,
            "color":         color,
            "total_analysts": total,
            "buy_pct":       round(buy_pct, 1),
            "hold_pct":      round(hold_pct, 1),
            "sell_pct":      round(sell_pct, 1),
            "counts": {"strong_buy": sb, "buy": b, "hold": h, "sell": s, "strong_sell": ss},
        }
    
    def _get_recent_moves(self, history: List[Dict]) -> Dict:
        """Find significant upgrades/downgrades in last 30 days."""
        recent = [r for r in history if r["days_ago"] <= 30]
        upgrades   = [r for r in recent if "UPGRADED" in r["action"]]
        downgrades = [r for r in recent if "DOWNGRADED" in r["action"]]
        initiations= [r for r in recent if "INITIATED" in r["action"]]
        
        return {
            "upgrades_30d":    len(upgrades),
            "downgrades_30d":  len(downgrades),
            "initiations_30d": len(initiations),
            "upgrade_list":    upgrades[:5],
            "downgrade_list":  downgrades[:5],
            "initiation_list": initiations[:5],
            "net_sentiment":   len(upgrades) - len(downgrades),
        }
    
    def _compute_rating_momentum(self, history: List[Dict]) -> Dict:
        """
        Rating momentum: is consensus improving or deteriorating?
        Compare average score last 30d vs 31-90d.
        Positive momentum = analysts becoming more bullish.
        """
        recent = [r["score"] for r in history if r["days_ago"] <= 30]
        older  = [r["score"] for r in history if 30 < r["days_ago"] <= 90]
        
        if not recent:
            return {"direction": "FLAT", "change": 0.0, "signal": "INSUFFICIENT DATA"}
        
        avg_recent = np.mean(recent)
        avg_older  = np.mean(older) if older else avg_recent
        delta      = avg_recent - avg_older
        
        if delta > 0.3:
            direction, signal = "IMPROVING ↑", "ANALYSTS TURNING BULLISH"
            color = "#00c896"
        elif delta < -0.3:
            direction, signal = "DETERIORATING ↓", "ANALYSTS TURNING BEARISH"
            color = "#ff4060"
        else:
            direction, signal = "STABLE →", "CONSENSUS UNCHANGED"
            color = "#e8b84b"
        
        return {
            "direction":    direction,
            "color":        color,
            "change":       round(delta, 2),
            "avg_recent":   round(avg_recent, 2),
            "avg_older":    round(avg_older, 2),
            "signal":       signal,
        }
    
    def _verdict(self, consensus: Dict, momentum: Dict, targets: Dict) -> Dict:
        """
        Final Wall Street verdict combining consensus + momentum + price target upside.
        
        Logic:
        - Start with consensus score
        - Adjust for momentum (improving = +0.2, deteriorating = -0.2)
        - Adjust for price target upside (>20% upside = +0.3)
        - Final verdict + confidence
        """
        score  = consensus.get("score", 3.0)
        m_adj  = 0.2 if momentum.get("change", 0) > 0.3 else -0.2 if momentum.get("change", 0) < -0.3 else 0
        
        # Price target upside
        mean_target  = targets.get("mean", 0)
        current      = targets.get("current", 0)
        target_adj   = 0.0
        upside_pct   = 0.0
        if mean_target > 0 and current > 0:
            upside_pct = (mean_target / current - 1) * 100
            target_adj = 0.3 if upside_pct > 20 else 0.2 if upside_pct > 10 else \
                        -0.3 if upside_pct < -10 else 0.0
        
        final = float(np.clip(score + m_adj + target_adj, 1, 5))
        
        # Confidence: based on analyst count and consensus spread
        n = consensus.get("total_analysts", 0)
        confidence = min(95, max(30, 50 + n * 1.5))
        
        # Verdict text
        if final >= 4.2:
            verdict = "WALL STREET LOVES IT"
            rec     = "Strong institutional tailwind. Majority of analysts bullish."
            icon    = "🟢"
        elif final >= 3.6:
            verdict = "BROADLY POSITIVE"
            rec     = "More buys than holds. Consensus favors long side."
            icon    = "🟡"
        elif final >= 2.8:
            verdict = "MIXED / NEUTRAL"
            rec     = "Analyst community divided. No strong directional consensus."
            icon    = "⚪"
        elif final >= 2.0:
            verdict = "CAUTIOUS"
            rec     = "More holds and sells. Institutional skepticism present."
            icon    = "🟠"
        else:
            verdict = "WALL STREET BEARISH"
            rec     = "Analysts predominantly negative. High conviction downside calls."
            icon    = "🔴"
        
        return {
            "verdict":      verdict,
            "icon":         icon,
            "score":        round(final, 2),
            "recommendation": rec,
            "confidence":   round(confidence),
            "upside_pct":   round(upside_pct, 1),
            "mean_target":  mean_target,
        }
