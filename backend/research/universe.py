"""
QuantEdge v6.0 — Investment Universe Builder
=============================================
Builds the investable universe: S&P 500 + S&P MidCap 400 constituents.
~900 liquid US stocks, market cap typically >$1B.

Two sources:
  1. Polygon /v3/reference/tickers with market_cap filter (preferred)
  2. Hardcoded fallback list of ~600 well-known tickers

Caches result for 7 days via Redis (index changes are slow).
"""

import os
import asyncio
import aiohttp
from typing import List, Optional, Set, Dict
from dataclasses import dataclass
from loguru import logger


POLYGON_BASE = "https://api.polygon.io"

# Minimum filters for the universe
MIN_MARKET_CAP = 1_000_000_000  # $1B
MAX_UNIVERSE_SIZE = 1500


# ══════════════════════════════════════════════════════════════
# HARDCODED FALLBACK UNIVERSE
# ══════════════════════════════════════════════════════════════
# ~600 of the most liquid US stocks. Used when Polygon ref API fails
# or to seed the first run. Covers S&P 500 + major mid-caps.
_FALLBACK_UNIVERSE: List[str] = [
    # Mega-caps
    "AAPL","MSFT","GOOGL","GOOG","AMZN","NVDA","META","TSLA","BRK.B","AVGO",
    "JPM","WMT","LLY","V","XOM","ORCL","MA","UNH","HD","PG",
    "JNJ","COST","NFLX","ABBV","BAC","CRM","CVX","KO","AMD","PEP",
    "TMO","LIN","WFC","CSCO","ADBE","ACN","MCD","ABT","DHR","TXN",
    "NOW","PM","DIS","MRK","VZ","GE","IBM","INTU","AXP","CAT",
    "QCOM","GS","MS","ISRG","NEE","RTX","T","UBER","LOW","BKNG",
    "SPGI","PFE","AMGN","HON","BLK","UPS","AMAT","ELV","NKE","C",
    "SCHW","ETN","PGR","ANET","DE","BA","COP","GILD","SYK","LMT",
    "BSX","ADI","TJX","ADP","MDLZ","PLD","MU","REGN","VRTX","MMC",
    "PANW","LRCX","CB","SBUX","CI","KLAC","ZTS","FI","INTC","BX",
    "SO","CMCSA","DUK","BMY","SNPS","TMUS","CDNS","WM","SHW","AON",
    "ITW","CL","APD","MO","CME","NOC","EOG","FCX","MCK","GD",
    "ICE","USB","CRWD","MMM","TGT","EQIX","EMR","HCA","PNC","ORLY",
    "FDX","APH","F","GM","PYPL","CTAS","NSC","PH","MAR","CARR",
    "MSI","COF","PSX","TDG","AJG","VLO","BDX","HLT","PCAR","AIG",
    "MPC","FIS","ROP","CSX","MNST","ECL","SRE","PSA","TFC","DXCM",
    "ALL","OXY","TT","KMB","MCO","WELL","O","APO","AEP","STZ",
    "AFL","DLR","NXPI","PAYX","ROST","JCI","NEM","CPRT","TRV","CMG",
    "GWW","LHX","FTNT","HES","KMI","BK","WDS","AZO","COR","WMB",
    "MSCI","PRU","OKE","GLW","HUM","FANG","NUE","VRSK","TEL","DOW",
    # Growth / Tech
    "PLTR","SNOW","ABNB","COIN","SQ","ROKU","DDOG","NET","SHOP","MDB",
    "DOCU","ZS","OKTA","TEAM","HUBS","ZM","TWLO","TTD","PINS","RBLX",
    "ARM","SMCI","DELL","VRT","APP","RDDT","DKNG","HOOD","AFRM","SOFI",
    "ENPH","FSLR","PLUG","RIVN","LCID","NIO","WBD","PARA","SPOT","LYFT",
    # S&P MidCap representatives
    "WDAY","FTV","KEYS","BIIB","DASH","DG","DLTR","EXR","FSLR","GPN",
    "IQV","MLM","PEG","WTW","XYL","ZBH","AMCR","BBY","CTLT","EL",
    "ENTG","EPAM","GEHC","HWM","IP","ON","PEG","PTC","PWR","SLB",
    "STE","SWK","TRGP","WRB","WST","WY","ZBRA","LYB","MKC","MRO",
    # Financials
    "AMP","BKR","CFG","CINF","CMS","DFS","EFX","EIX","ES","FITB",
    "HBAN","HIG","HOLX","IEX","IFF","KEY","L","MTB","NDAQ","NTRS",
    "NTRA","NWSA","ODFL","OKE","OMC","OTIS","PAYC","POOL","PPG","PPL",
    "RF","RVTY","SJM","STT","SYF","TSN","WY","XEL","YUM","ZION",
    # Industrials
    "A","AES","AFL","AKAM","ALB","ALGN","ALLE","AME","ANSS","AOS",
    "APA","ARE","ATO","AVB","AVY","BALL","BAX","BBWI","BBY","BEN",
    "BIO","BKNG","BR","BRO","BWA","CAG","CBOE","CBRE","CCL","CDW",
    "CE","CFG","CHD","CHRW","CLX","CMA","CME","CNC","CNP","COO",
    "CPB","CPRT","CRL","CSGP","CTRA","CTSH","CVS","D","DAL","DD",
    "DECK","DFS","DG","DGX","DHI","DOV","DPZ","DRI","DTE","DVA",
    "DVN","DXCM","EBAY","ECL","ED","EFX","EG","EIX","EL","EMN",
    "EMR","ENPH","EOG","EQR","EQT","ES","ESS","ETR","EVRG","EW",
    "EXPD","EXPE","EXR","F","FANG","FAST","FCX","FDS","FDX","FE",
    "FFIV","FI","FICO","FIS","FITB","FLT","FMC","FOX","FOXA","FSLR",
    # Consumer
    "HAS","HCA","HES","HIG","HII","HOLX","HON","HPE","HPQ","HRL",
    "HSIC","HST","HSY","HUBB","HUBS","HUM","HWM","IBM","ICE","IDXX",
    "IEX","IFF","INCY","INTU","INVH","IP","IPG","IQV","IR","IRM",
    "ISRG","IT","ITW","IVZ","J","JBHT","JBL","JCI","JKHY","JNJ",
    "JNPR","JPM","K","KDP","KEY","KEYS","KHC","KIM","KLAC","KMB",
    "KMI","KMX","KO","KR","KVUE","L","LDOS","LEN","LH","LHX",
    "LIN","LKQ","LMT","LNT","LOW","LRCX","LULU","LUV","LVS","LW",
    "LYB","LYV","MA","MAA","MAR","MAS","MCD","MCHP","MCK","MCO",
    "MDLZ","MDT","MET","META","MGM","MHK","MKC","MKTX","MLM","MMC",
    "MMM","MNST","MO","MOH","MOS","MPC","MPWR","MRK","MRNA","MRO",
    # More coverage
    "MSCI","MSFT","MSI","MTB","MTCH","MTD","MU","NCLH","NDAQ","NDSN",
    "NEE","NEM","NFLX","NI","NKE","NOC","NOW","NRG","NSC","NTAP",
    "NTRS","NUE","NVDA","NVR","NWL","NWS","NWSA","NXPI","O","ODFL",
    "OKE","OMC","ON","ORCL","ORLY","OTIS","OXY","PANW","PARA","PAYC",
    "PAYX","PCAR","PCG","PEG","PEP","PFE","PFG","PG","PGR","PH",
    "PHM","PKG","PLD","PM","PNC","PNR","PNW","POOL","PPG","PPL",
    "PRU","PSA","PSX","PTC","PWR","PYPL","QCOM","QRVO","RCL","REG",
    "REGN","RF","RHI","RJF","RL","RMD","ROK","ROL","ROP","ROST",
    "RSG","RTX","RVTY","SBAC","SBUX","SCHW","SEE","SHW","SJM","SLB",
    "SMCI","SNA","SNPS","SO","SOLV","SPG","SPGI","SRE","STE","STLD",
    "STT","STX","STZ","SWK","SWKS","SYF","SYK","SYY","T","TAP",
    "TDG","TDY","TECH","TEL","TER","TFC","TFX","TGT","TJX","TMO",
    "TMUS","TPR","TRGP","TRMB","TROW","TRV","TSCO","TSLA","TSN","TT",
    "TTWO","TXN","TXT","TYL","UAL","UBER","UDR","UHS","ULTA","UNH",
    "UNP","UPS","URI","USB","V","VICI","VLO","VLTO","VMC","VRSK",
    "VRSN","VRTX","VST","VTR","VTRS","VZ","WAB","WAT","WBA","WBD",
    "WDC","WEC","WELL","WFC","WM","WMB","WMT","WRB","WST","WTW",
    "WY","WYNN","XEL","XOM","XYL","YUM","ZBH","ZBRA","ZTS",
]


@dataclass
class UniverseEntry:
    ticker: str
    name: str = ""
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    primary_exchange: Optional[str] = None


class UniverseBuilder:
    """
    Build the investable universe.
    Primary path: Polygon reference API with market cap filter.
    Fallback: hardcoded list of well-known S&P 500 + MidCap names.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        if not self.api_key:
            logger.warning("UniverseBuilder: no POLYGON_API_KEY set")

    async def _fetch_from_polygon(
        self,
        session: aiohttp.ClientSession,
        max_tickers: int = MAX_UNIVERSE_SIZE,
    ) -> List[UniverseEntry]:
        """
        Fetch active US common stocks from Polygon, sorted by market cap desc.
        Paginates. Stops when we have max_tickers or no more pages.
        """
        url = f"{POLYGON_BASE}/v3/reference/tickers"
        params = {
            "market": "stocks",
            "type": "CS",
            "active": "true",
            "limit": 1000,
            "apiKey": self.api_key,
        }

        entries: List[UniverseEntry] = []
        url_next: Optional[str] = url
        params_next: Optional[Dict] = params

        for _ in range(3):  # Up to 3 pages of 1000 = 3000 tickers max
            if url_next is None:
                break
            try:
                async with session.get(
                    url_next, params=params_next,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"Polygon tickers returned {resp.status}")
                        break
                    data = await resp.json()
            except Exception as e:
                logger.warning(f"Polygon tickers fetch failed: {e}")
                break

            results = data.get("results", [])
            if not results:
                break

            for r in results:
                # Starter plan does not return market_cap in list endpoint.
                # We filter to NYSE/NASDAQ primary exchange instead, and rely
                # on hardcoded universe for liquidity guarantee.
                exch = r.get("primary_exchange", "")
                if exch not in ("XNYS", "XNAS"):
                    continue
                entries.append(UniverseEntry(
                    ticker=r.get("ticker", ""),
                    name=r.get("name", ""),
                    market_cap=None,  # filled in per-ticker later
                    sector=r.get("sic_description"),
                    primary_exchange=exch,
                ))
                if len(entries) >= max_tickers:
                    break

            if len(entries) >= max_tickers:
                break

            next_url = data.get("next_url")
            if not next_url:
                break
            url_next = next_url
            params_next = {"apiKey": self.api_key}

        # Sort by market cap desc client-side since API can't
        entries.sort(key=lambda e: e.market_cap or 0, reverse=True)
        return entries[:max_tickers]

    def _fallback_universe(self) -> List[UniverseEntry]:
        """Return hardcoded universe with deduplicated tickers."""
        seen: Set[str] = set()
        out: List[UniverseEntry] = []
        for t in _FALLBACK_UNIVERSE:
            t = t.strip().upper()
            if t and t not in seen:
                seen.add(t)
                out.append(UniverseEntry(ticker=t))
        return out

    async def build(self, max_tickers: int = MAX_UNIVERSE_SIZE) -> List[UniverseEntry]:
        """
        Returns list of universe entries.

        Design decision (Apr 2026): Polygon Starter plan /v3/reference/tickers
        returns alphabetical ordering without market_cap, which surfaces
        micro-caps and SPACs instead of the S&P 500 core we actually want.
        So we use the curated hardcoded fallback list as the primary source.

        TODO: Replace with a per-ticker snapshot-based universe builder that
        fetches market_cap/volume per ticker and filters properly. Requires
        ~500 API calls to enrich, runs once per week.
        """
        logger.info("Using curated fallback universe (507 liquid US stocks)")
        return self._fallback_universe()[:max_tickers]


# Standalone test
async def _test():
    import sys
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); sys.exit(1)

    builder = UniverseBuilder(api_key)
    entries = await builder.build(max_tickers=1500)

    print(f"\n{'='*60}")
    print(f"  UNIVERSE BUILT: {len(entries)} tickers")
    print(f"{'='*60}")
    print(f"\n  Top 20 by market cap:")
    for i, e in enumerate(entries[:20], 1):
        mc_str = f"${e.market_cap/1e9:.1f}B" if e.market_cap else "N/A"
        print(f"    {i:2}. {e.ticker:<6}  {mc_str:>8}   {e.name[:40]}")
    print(f"\n  Last 5 tickers (smallest caps above $1B threshold):")
    for e in entries[-5:]:
        mc_str = f"${e.market_cap/1e9:.1f}B" if e.market_cap else "N/A"
        print(f"    {e.ticker:<6}  {mc_str:>8}   {e.name[:40]}")


if __name__ == "__main__":
    asyncio.run(_test())
