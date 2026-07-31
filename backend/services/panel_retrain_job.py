"""Nightly panel rebuild + retrain.

Runs the full-universe panel build and the multi-horizon trainer as SUBPROCESSES,
into a STAGING directory, and promotes the result only if it validates.

Both properties are deliberate. Running in-process would leave the panel's
peak memory resident in the API worker for the rest of the day on a 4GB host.
Training straight into the live model directory is how the previous 707-ticker
models were lost: the trainer picked the newest panel on disk, which happened to
be an older 80-ticker file, and overwrote every artifact with no way back.
"""
from __future__ import annotations
import asyncio, json, os, shutil, time
from datetime import datetime
from pathlib import Path
from loguru import logger

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/models"))
PANEL_DIR = MODEL_DIR / "panels"
LIVE_DIR = MODEL_DIR / "panel"
STAGE_ROOT = MODEL_DIR / "_staging"
KEEP_PANELS = 4
KEEP_ROLLBACKS = 3

HORIZONS = ["5", "10", "21", "63", "126", "252"]


class PanelRetrainJob:
    def __init__(self, tickers: int = 707, years: int = 5,
                 build_timeout_s: int = 5 * 3600, train_timeout_s: int = 3600):
        self.tickers = tickers
        self.years = years
        self.build_timeout_s = build_timeout_s
        self.train_timeout_s = train_timeout_s

    async def _run(self, args: list[str], env: dict, timeout_s: int, tag: str) -> bool:
        logger.info(f"[retrain] {tag}: {' '.join(args)}")
        proc = await asyncio.create_subprocess_exec(
            *args, cwd="/app", env={**os.environ, **env},
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            proc.kill()
            logger.error(f"[retrain] {tag} exceeded {timeout_s}s — killed")
            return False
        text = (out or b"").decode(errors="replace")
        for line in text.strip().splitlines()[-12:]:
            logger.info(f"[retrain] {tag} | {line}")
        if proc.returncode != 0:
            logger.error(f"[retrain] {tag} exited {proc.returncode}")
            return False
        return True

    def _newest_panel(self) -> Path | None:
        ps = sorted(PANEL_DIR.glob("panel_*.parquet"))
        return ps[-1] if ps else None

    def _validate(self, report_path: Path, min_tickers: int) -> tuple[bool, str]:
        """A staged model set is promoted only if the report is complete and the
        universe is not smaller than what is already live. A silent shrink from
        707 tickers to 80 is what made the previous models unusable."""
        if not report_path.exists():
            return False, "no training_report.json"
        try:
            r = json.loads(report_path.read_text())
        except Exception as e:
            return False, f"unreadable report: {e}"
        n = int(r.get("n_tickers") or 0)
        if n < min_tickers:
            return False, f"universe shrank to {n} tickers (live has {min_tickers})"
        hz = r.get("horizons") or {}
        missing = [h for h in HORIZONS if h not in hz]
        if missing:
            return False, f"missing horizons {missing}"
        for h in HORIZONS:
            for f in (f"xgb_{h}d.joblib", f"lgb_{h}d.joblib"):
                if not (report_path.parent / f).exists():
                    return False, f"missing artifact {f}"
        return True, f"{n} tickers, {len(hz)} horizons"

    def _live_ticker_count(self) -> int:
        p = LIVE_DIR / "training_report.json"
        if not p.exists():
            return 0
        try:
            return int(json.loads(p.read_text()).get("n_tickers") or 0)
        except Exception:
            return 0

    def _prune(self):
        panels = sorted(PANEL_DIR.glob("panel_*.parquet"))
        for old in panels[:-KEEP_PANELS]:
            meta = old.with_name(old.stem + "_meta.json")
            old.unlink(missing_ok=True); meta.unlink(missing_ok=True)
            logger.info(f"[retrain] pruned old panel {old.name}")
        backups = sorted(MODEL_DIR.glob("panel_prev_*"))
        for old in backups[:-KEEP_ROLLBACKS]:
            shutil.rmtree(old, ignore_errors=True)
            logger.info(f"[retrain] pruned old rollback {old.name}")

    async def run(self):
        t0 = time.time()
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        logger.info(f"[retrain] START universe={self.tickers} years={self.years}")

        before = self._newest_panel()
        ok = await self._run(
            ["python", "-m", "ml.training.build_panel", "--full",
             "--tickers", str(self.tickers), "--years", str(self.years)],
            {"MODEL_DIR": str(MODEL_DIR)}, self.build_timeout_s, "build_panel")
        if not ok:
            logger.error("[retrain] ABORT — panel build failed; live models untouched")
            return
        newest = self._newest_panel()
        if newest is None or (before is not None and newest == before):
            logger.error("[retrain] ABORT — build produced no new panel; live models untouched")
            return
        logger.info(f"[retrain] panel ready: {newest.name}")

        # Train into staging against ONLY the panel just built, so the trainer
        # cannot pick up an older/smaller file the way it did before.
        stage = STAGE_ROOT / stamp
        shutil.rmtree(STAGE_ROOT, ignore_errors=True)
        (stage / "panels").mkdir(parents=True, exist_ok=True)
        shutil.copy2(newest, stage / "panels" / newest.name)
        ok = await self._run(
            ["python", "-m", "ml.training.train_panel_models"],
            {"MODEL_DIR": str(stage)}, self.train_timeout_s, "train_panel")
        if not ok:
            logger.error("[retrain] ABORT — training failed; live models untouched")
            shutil.rmtree(STAGE_ROOT, ignore_errors=True)
            return

        staged_report = stage / "panel" / "training_report.json"
        good, why = self._validate(staged_report, min_tickers=max(1, int(self._live_ticker_count() * 0.9)))
        if not good:
            logger.error(f"[retrain] ABORT — staged models rejected: {why}; live models untouched")
            shutil.rmtree(STAGE_ROOT, ignore_errors=True)
            return
        logger.info(f"[retrain] staged models validated: {why}")

        # Promote: keep the outgoing set so a bad promotion is one mv from undone.
        if LIVE_DIR.exists():
            rollback = MODEL_DIR / f"panel_prev_{stamp}"
            shutil.rmtree(rollback, ignore_errors=True)
            LIVE_DIR.rename(rollback)
            logger.info(f"[retrain] previous models kept at {rollback.name}")
        (stage / "panel").rename(LIVE_DIR)
        shutil.rmtree(STAGE_ROOT, ignore_errors=True)

        try:
            from ml.serving.panel_predictor import PanelPredictor
            PanelPredictor._instance = None
            p = PanelPredictor.get()
            logger.info(f"[retrain] predictor reloaded, available={p.available()}")
        except Exception as e:
            logger.error(f"[retrain] predictor reload failed: {e}")

        self._prune()
        logger.info(f"[retrain] DONE in {(time.time()-t0)/60:.1f} min")
