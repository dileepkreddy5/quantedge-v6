# QuantEdge — Handoff

Last rewritten 2026-09-27. Supersedes the August handoff, which predates
Pattern Lab, Company Intelligence, the rebound resurrection and the disk outage.

## Working agreement
- Server: `ssh root@178.156.190.252`, app at `/opt/quantedge`, branch `tab-audit`.
  Mac clone `~/Desktop/quantedge_v6`, sync with `git pull --ff-only` after every push.
- Backend code is COPIED into the image, not mounted: every host edit needs
  `docker compose build backend && docker compose up -d`. Frontend likewise
  (`build frontend`); CRA fails the build on any TS error.
- **A backend rebuild recreates the API container and kills anything exec'd
  inside it.** Detached scans, library builds and ingests die silently.
  Never rebuild while one is running. Killed builds this way four times.
- An exec'd process survives a dropped SSH session (it lives in the container),
  but NOT a container rebuild. Still: run long jobs detached to a log file
  (`docker compose exec -d backend sh -c '... > /app/models/X.log 2>&1'`) so the
  output survives too. Kill a stray one from the host: `docker top quantedge-api`
  gives host PIDs, then `kill <pid>`.
- Manual runners live in `backend/run_*.py` (in the repo, so they survive
  rebuilds): `build_lib`, `run_formations`, `run_conditions`, `run_rebound`,
  `run_ci_ingest`, `run_ci_capital`. Run with
  `docker compose exec -T backend python -u /app/run_X.py`. Use `-u` and
  no `tail` if you want to see progress live.
- Methodology: no fake values. Read files whole (grep-and-guess caused
  bugs). Verify claimed-scheduled jobs actually produce output. Commit
  full builds, not phases.

## Infrastructure (Hetzner CPX21, 3 vCPU / 3.7GB / 75GB)
- Five containers: caddy (public), frontend (nginx), api (FastAPI),
  postgres, redis. Compose service for the DB is `postgres`
  (container `quantedge-db`). All `restart: unless-stopped`.
- Volumes: `postgres_data`, `model_data:/app/models` (trained models,
  pattern libraries, scan logs), `edgar_data:/app/data` (artifacts,
  companyfacts.zip, rebound artifact), caddy certs.
- **Disk outage 2026-09-17→27:** a cron tarred the 3.5GB model volume
  daily with 14-day retention (36GB) plus unretained pg dumps filled the
  disk; Postgres rejected connections; every nightly job failed silently
  for ten nights. Fixed: retention (pg 7d, model 2d), Docker log rotation
  (50m×3) on all services, `/system/stats` reports disk with a warning
  above 85% and SystemBar shows it. `df -h /` is the first thing to check
  on any cold start.
- Memory is tight (swap in use). A reboot is pending (`*** System restart
  required ***`). Do it only outside the nightly window (see schedule).
- Secrets in `/opt/quantedge/.env` (root-only). Double `$` in bcrypt hashes.

## Nightly schedule (all jobs registered in main_v6 lifespan)
| ET | Job | Output | How to verify |
|---|---|---|---|
| 02:00 | multibagger scan (re-downloads companyfacts.zip first) | `/app/data/artifacts/scan_artifact.json` | `/api/v6/system/stats` boards age |
| 02:15 MT | panel retrain (707 tickers, 5y) | `/app/models/panel/training_report.json` | report `trained_at`, `n_train` differs per horizon |
| 02:30 | rebound scan v1 | `/app/data/rebound_artifact.json` | `/api/v6/rebound/list` tiers |
| 03:30 | relationship_extract | `cf_artifact.json` — **frozen since July; write path never audited** | stats shows stale |
| 05:00 | Company Intelligence ingest (EDGAR 8-K/Form 4, then XBRL capital) | `ci_*` tables | row counts grow; job logs ✅/❌ |
| 17:30 | bars sync (one session; does NOT backfill gaps) | `daily_bars` | `SELECT max(d)` |
| 18:15 | peer scan | peer_stats | — |
| 18:45 | pattern rebuild: library + formations + conditions | `/app/models/patterns/*.npz`, artifacts | file mtimes |
| 08:00 UTC | cron pg_dumpall → /root/backups (7d retention) | | |
| 09:00 UTC | cron model volume tar → /root/model_backups (2d retention) | | |

If the box is down for N days, `daily_bars` has an N-day hole the nightly
sync will not fill. Catch up with a loop over `BarsStore.sync_day(date)`
(see the Sep 27 fix). Then rebuild the pattern library.

## Data
- `daily_bars`: full US universe, 5 years (2021-08-30→), 3.5M+ rows.
  Polygon Stocks Starter = 5y, US only. There is no international data.
- `universe`: ticker, name, SIC, CIK, market_cap. CIKs are 10-digit padded.
- `signals` (~300 rows): per-analysis model outputs with realized
  `ret_5d/21d/63d` filled by the outcome filler. `xgb_confidence` and
  `recommended_position` are null in recent rows — signal-writer bug, open.
- `holdings_13f`, `peer_stats`, `relationships`, `ascent_snapshots`.
- `ci_raw_evidence` / `ci_events` / `ci_derived` / `ci_entity_map`:
  Company Intelligence, append-only (see below).
- companyfacts.zip (1.4GB, SEC XBRL bulk) on the edgar volume; the
  multibagger, capital-allocation adapter and peer fundamentals read it.
  `download_bulk` now rejects files under 500MB and writes atomically.

## Subsystems
**Panel models** (`backend/ml/training/`): XGBoost+LightGBM per horizon
(5/10/21/63/126/252d), nightly. IC gate: one per-date IC series on the
scoring half, HAC t-stat, `ic_independent` null under 5 non-overlapping
windows. Per-horizon labels (the 252d truncation bug is fixed; `n_train`
must differ per horizon in the report). Validated horizons as of Aug 29:
2wk (t +3.58) and 1mo (t +2.69). Manual vs nightly reports use different
date grids and are not directly comparable.

**Multibagger / Ascent / Rebound boards**: nightly artifacts on the edgar
volume via `core/artifact_paths.py`. Rebound v1 (`quantedge/fundamentals/
rebound/scan.py`) was built from scratch on Aug 30 — the original engine
never existed in the repo. No insider dimension.

**Pattern Lab** (`quantedge/patterns/`, tab 🧬): analog library (538k
20d / 516k 60d windows, z-scored, six horizons, SPY excess, vol/52w
context), two-stage matching (z-Euclid → banded DTW), episode dedup,
conditional filters, state vector, forward fan from real forward closes;
formations (LMW-style, 58k occurrences with regime/volume/vol context);
conditions (394k samples, quintiles) with a 9-state transition matrix.
Motif discovery deliberately absent. Loop bounds must use
`min(HORIZONS)` — bounding by max silently drops the most recent windows.

**Company Intelligence** (`quantedge/intel/`, tab 🔎) — Phase A complete 2026-09-27: frozen contract —
`ci_raw_evidence` immutable (accession + hash, `supersedes_id`),
`ci_events` OBSERVED with deterministic significance, `ci_derived`
DERIVED/HYPOTHESIS with `extractor_version` and structured citations,
`ci_entity_map` with match provenance. **No UPDATE/DELETE in ingest code.**
`available_at` = SEC acceptance timestamp (or first-report filing date for
XBRL), never the economic `event_date`. Adapters live: EDGAR 8-K/Form 4 (A1/A2), XBRL capital
allocation (A3, periods anchored to their FIRST reporting filing),
13F institutional (A4: complete quarters only, >=2000 managers;
available_at = quarter end + 45d; resolves ticker->CUSIP via the OWNERSHIP
tab's ownership_for — one resolver — cached in ci_entity_map), market
attention (A5: one pass/day over all Polygon news; spikes need a full 97-day
baseline). Universe: all active filers. Next: patents,
research, talent, products (B), USAspending/openFDA/ClinicalTrials (C),
FRED industry state (D). Customers, capacity, transcripts: NO SOURCE —
displayed as such, never inferred. SEC pace: 2 workers, ~3 req/s;
3 workers drew a block.

**Homepage**: Daily Brief (`/api/v6/brief/today` from `signals`), index
proxies (SPY/QQQ/XLK via Polygon /prev, uncached), SystemBar
(`/api/v6/system/stats`: live counts, board staleness, disk), boards.

## Open items
1. Panel expansion beyond 707: feature build is 7.2s/ticker after the Sep 27
   vectorization; a 3-process pool makes all filers a ~2.7h nightly build.
   Structural refactor (compute indicators once per series, not per sample)
   is worth up to another 10x. Observe the first DB-path nightly first.
2. recommended_position and vol_scale both 0.2 in the first populated signal
   row — check for a floor/cap in portfolio_construction.
3. CI Phase B/C/D adapters per the contract; OpenFIGI if a second CUSIP
   source is ever needed.
5. Redis cache for `/brief/indices` if traffic ever warrants.
6. AI agent (tool-calling over QuantEdge endpoints; report, never invent;
   rate-limited or login-gated) — deliberately last.
7. Cleanups across older tabs.
