"""Bulk EDGAR loader — the whole market's facts in one download, no throttling.

Per-company API calls get rate-limited (429) when scanning thousands. EDGAR
publishes companyfacts.zip: EVERY company's full XBRL facts in one file,
refreshed nightly. Download once (~1.2GB), then read any company's fundamentals
from local disk with zero API calls. This is how you scan the full universe.

Usage:
  download_bulk()                      # one-time / nightly, ~1.2GB
  facts = company_facts_from_bulk(cik) # local read, instant, no rate limit
"""
from __future__ import annotations
import os, json, zipfile, urllib.request, shutil

UA = "Dileep Kapu dileepkreddy5@gmail.com"
BULK_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "edgar_bulk")
ZIP_PATH = os.path.join(BULK_DIR, "companyfacts.zip")
BULK_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"


def download_bulk(force=False):
    """Download EDGAR's full company-facts archive (~1.2GB). One call, no per-company hits."""
    os.makedirs(BULK_DIR, exist_ok=True)
    if os.path.exists(ZIP_PATH) and not force:
        mb = os.path.getsize(ZIP_PATH) / 1e6
        print(f"bulk zip already present ({mb:.0f} MB) — use force=True to refresh")
        return ZIP_PATH
    print("downloading EDGAR companyfacts.zip (~1.2GB, a few minutes)…", flush=True)
    req = urllib.request.Request(BULK_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=600) as r, open(ZIP_PATH, "wb") as f:
        shutil.copyfileobj(r, f)
    mb = os.path.getsize(ZIP_PATH) / 1e6
    print(f"downloaded {mb:.0f} MB -> {ZIP_PATH}", flush=True)
    return ZIP_PATH


def company_facts_from_bulk(cik: str) -> dict | None:
    """Read one company's facts from the bulk zip (no network). cik = 10-digit."""
    if not os.path.exists(ZIP_PATH):
        raise FileNotFoundError("bulk zip not downloaded — run download_bulk() first")
    member = f"CIK{cik}.json"
    try:
        with zipfile.ZipFile(ZIP_PATH) as z:
            with z.open(member) as fh:
                return json.load(fh)
    except KeyError:
        return None  # company not in archive
    except Exception:
        return None


if __name__ == "__main__":
    download_bulk()
