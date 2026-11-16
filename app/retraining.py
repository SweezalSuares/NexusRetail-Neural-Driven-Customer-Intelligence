"""Batch-triggered model retraining for mature local customer orders."""
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime

from src import config as C
from src.features import load_data

from app import accounts

MINIMUM_NEW_MATURE_ORDERS = 100


def maybe_start_retraining():
    matured_count = accounts.get_matured_order_count()
    return_count = accounts.get_matured_return_count()
    if not accounts.claim_retraining(
        matured_count, return_count, MINIMUM_NEW_MATURE_ORDERS
    ):
        return False

    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = C.OUT_DIR / "retraining.log"
    try:
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.Popen(
                [sys.executable, "-m", "app.retraining"],
                cwd=C.ROOT,
                env=os.environ.copy(),
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
    except OSError as exc:
        accounts.update_state({
            "training_status": "failed",
            "training_error": f"Could not start the retraining process: {exc}",
        })
        return False
    return True


def run_worker():
    if accounts.get_state("training_status") != "queued":
        return
    accounts.update_state({
        "training_status": "running",
        "training_started_at": datetime.now().astimezone().isoformat(),
        "training_error": "",
    })
    version = uuid.uuid4().hex
    models_dir = C.MODELS_DIR / "versions" / version
    reports_dir = C.REPORTS_DIR / "versions" / version
    scores_dir = C.OUT_DIR / "score_versions" / version
    try:
        from src.pipeline import load_models, score_customers, train_all

        matured_count = accounts.get_matured_order_count()
        return_count = accounts.get_matured_return_count()
        data = load_data(include_live=True)
        active_version = accounts.get_state("model_version")
        fallback_models = (
            C.MODELS_DIR / "versions" / active_version if active_version else C.MODELS_DIR
        )
        fallback_reports = (
            C.REPORTS_DIR / "versions" / active_version if active_version else C.REPORTS_DIR
        )
        train_all(
            data, models_dir=models_dir, reports_dir=reports_dir,
            fallback_models_dir=fallback_models, fallback_reports_dir=fallback_reports,
        )
        scores, diagnostics = score_customers(
            data, load_models(models_dir), output_dir=scores_dir
        )
        metrics_path = reports_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["meta"].update(diagnostics)
        metrics["meta"]["model_version"] = version
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        accounts.update_state({
            "model_version": version,
            "last_matured_count": matured_count,
            "last_return_event_count": return_count,
            "training_status": "complete",
            "training_completed_at": datetime.now().astimezone().isoformat(),
            "training_error": "",
        })
        print(f"Retraining completed: version={version}, customers={len(scores)}", flush=True)
    except Exception as exc:
        accounts.update_state({
            "training_status": "failed",
            "training_error": f"{type(exc).__name__}: {exc}",
        })
        raise


if __name__ == "__main__":
    run_worker()
