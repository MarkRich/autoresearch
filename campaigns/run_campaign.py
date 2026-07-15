#!/usr/bin/env python3
"""Run a finite, paired two-GPU autoresearch campaign.

Each round sends the same experiment to both GPU lanes with different fixed
seeds. This makes comparisons about repeatable changes, not one lucky run.
The default plan is six 30-minute rounds: roughly three hours.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DEFAULT_BUDGET_SECONDS = 1800
SEEDS = (101, 202)

BASE_ENV = {
    "DEVICE_BATCH_SIZE": "2",
    "TOTAL_BATCH_SIZE": str(2**19),
    "AMP_DTYPE": "fp16",
    "OPENCLAW_FORCE_SDPA": "1",
    "OPENCLAW_DISABLE_TORCH_COMPILE": "1",
    "WINDOW_PATTERN": "LLLL",
    "DEPTH": "8",
    "ASPECT_RATIO": "64",
    "HEAD_DIM": "128",
    "USE_VALUE_EMBEDS": "1",
    "OPTIMIZER_KIND": "muon",
    "EMBEDDING_LR": "0.010",
    "UNEMBEDDING_LR": "0.002",
    "MATRIX_LR": "0.0020",
    "SCALAR_LR": "0.010",
    "WEIGHT_DECAY": "0.01",
    "ADAM_BETA1": "0.8",
    "ADAM_BETA2": "0.95",
    "WARMUP_RATIO": "0.15",
    "WARMDOWN_RATIO": "0.60",
    "FINAL_LR_FRAC": "0.05",
    "TRAIN_PROBE_BATCHES": "4",
    # Observe regressions in telemetry but let every finite run finish.
    "FAILFAST_MIN_PROGRESS": "2.0",
    "FAILFAST_REGRESSION_MIN_RISE": "1000",
}

PLAN = (
    ("control", {}),
    ("no_value_embeddings", {"USE_VALUE_EMBEDS": "0"}),
    ("matrix_lr_0015", {"MATRIX_LR": "0.0015"}),
    ("matrix_lr_0025", {"MATRIX_LR": "0.0025"}),
    ("warmdown_70", {"WARMDOWN_RATIO": "0.70"}),
    ("zero_final_lr", {"FINAL_LR_FRAC": "0.0"}),
)


@dataclass
class ChildRun:
    gpu: int
    run_id: str
    log_path: Path
    metric_path: Path
    process: subprocess.Popen[Any]
    log_handle: Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def load_dashboard_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in {"DASHBOARD_INGEST_URL", "DASHBOARD_INGEST_TOKEN", "DASHBOARD_SITES_AUTH_TOKEN"}:
            result[key] = value
    return result


def read_final(metric_path: Path) -> dict[str, Any] | None:
    if not metric_path.exists():
        return None
    final = None
    for line in metric_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("kind") == "final":
            final = event
    return final


def post_event(event: dict[str, Any], dashboard: dict[str, str]) -> None:
    url = dashboard.get("DASHBOARD_INGEST_URL")
    if not url:
        return
    headers = {"content-type": "application/json"}
    if dashboard.get("DASHBOARD_INGEST_TOKEN"):
        headers["authorization"] = f"Bearer {dashboard['DASHBOARD_INGEST_TOKEN']}"
    if dashboard.get("DASHBOARD_SITES_AUTH_TOKEN"):
        headers["oai-sites-authorization"] = f"Bearer {dashboard['DASHBOARD_SITES_AUTH_TOKEN']}"
    request = urllib.request.Request(
        url,
        data=json.dumps(event).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
    except Exception as exc:
        print(f"telemetry warning: {type(exc).__name__}: {exc}", flush=True)


def control_event(campaign_id: str, kind: str, **payload: Any) -> dict[str, Any]:
    return {
        "campaign_id": campaign_id,
        "run_id": f"{campaign_id}-control",
        "kind": kind,
        "time": time.time(),
        "experiment_label": "campaign",
        **payload,
    }


def launch_pair(
    campaign_id: str,
    round_number: int,
    label: str,
    overrides: dict[str, str],
    budget_seconds: int,
    python: str,
    dashboard: dict[str, str],
    code_sha: str,
    root: Path,
) -> list[ChildRun]:
    logs = root / "logs"
    metrics = root / "metrics"
    logs.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    children: list[ChildRun] = []
    for gpu, seed in enumerate(SEEDS):
        run_id = f"{campaign_id}-r{round_number:02d}-gpu{gpu}-{label}-s{seed}"
        log_path = logs / f"{run_id}.log"
        metric_path = metrics / f"{run_id}.jsonl"
        env = os.environ.copy()
        env.update(BASE_ENV)
        env.update(overrides)
        env.update(dashboard)
        env.update({
            "CODEX_WORK_TAG": "autoresearch-30m-campaign",
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "TIME_BUDGET_SECONDS": str(budget_seconds),
            "AUTORESEARCH_RUN_ID": run_id,
            "AUTORESEARCH_CAMPAIGN_ID": campaign_id,
            "AUTORESEARCH_CAMPAIGN_ROUND": str(round_number),
            "AUTORESEARCH_GPU_INDEX": str(gpu),
            "AUTORESEARCH_SEED": str(seed),
            "AUTORESEARCH_CODE_SHA": code_sha,
            "AUTORESEARCH_METRICS_DIR": str(metrics),
            "EXPERIMENT_LABEL": label,
        })
        log_handle = log_path.open("a", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            [python, "-u", "train.py"], cwd=REPO, env=env,
            stdout=log_handle, stderr=subprocess.STDOUT, start_new_session=True,
        )
        children.append(ChildRun(gpu, run_id, log_path, metric_path, process, log_handle))
        print(f"launched gpu={gpu} pid={process.pid} run={run_id}", flush=True)
    return children


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def terminate_children(children: list[ChildRun]) -> None:
    for child in children:
        if child.process.poll() is None:
            try:
                os.killpg(child.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 20
    while time.time() < deadline and any(child.process.poll() is None for child in children):
        time.sleep(0.5)
    for child in children:
        if child.process.poll() is None:
            try:
                os.killpg(child.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-seconds", type=int, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--rounds", type=int, default=len(PLAN))
    parser.add_argument("--campaign-id", default=f"codex-30m-{utc_stamp().lower()}")
    parser.add_argument("--python", default="/home/mark/autoresearch-transformer-paper/.venv/bin/python")
    parser.add_argument("--dashboard-env", type=Path, default=Path.home() / ".config/autoresearch-dashboard.env")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.budget_seconds < 30:
        raise SystemExit("budget must be at least 30 seconds")
    round_plan = PLAN[: max(1, min(args.rounds, len(PLAN)))]
    campaign_root = REPO / "campaign-results" / args.campaign_id
    state_path = campaign_root / "state.json"
    dashboard = load_dashboard_env(args.dashboard_env)
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    end_estimate = time.time() + len(round_plan) * (args.budget_seconds + 180)
    state: dict[str, Any] = {
        "campaign_id": args.campaign_id, "status": "running",
        "budget_seconds": args.budget_seconds, "rounds_total": len(round_plan),
        "code_sha": code_sha, "started_at": time.time(),
        "estimated_end_at": end_estimate, "runs": [],
    }
    write_state(state_path, state)
    post_event(control_event(
        args.campaign_id, "campaign_start", budget_seconds=args.budget_seconds,
        rounds_total=len(round_plan), estimated_end_at=end_estimate, code_sha=code_sha,
    ), dashboard)

    active: list[ChildRun] = []
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    try:
        for round_number, (label, overrides) in enumerate(round_plan, start=1):
            if stopping:
                break
            post_event(control_event(args.campaign_id, "round_start", round=round_number, experiment_label=label), dashboard)
            active = launch_pair(
                args.campaign_id, round_number, label, overrides, args.budget_seconds,
                args.python, dashboard, code_sha, campaign_root,
            )
            hard_deadline = time.time() + args.budget_seconds + 600
            while not stopping and any(child.process.poll() is None for child in active):
                if time.time() > hard_deadline:
                    print(f"round {round_number} exceeded hard deadline; terminating", flush=True)
                    terminate_children(active)
                    break
                state["active_round"] = round_number
                state["active_experiment"] = label
                state["active_runs"] = [
                    {"gpu": child.gpu, "run_id": child.run_id, "pid": child.process.pid,
                     "returncode": child.process.poll()} for child in active
                ]
                write_state(state_path, state)
                time.sleep(15)
            if stopping:
                terminate_children(active)
            round_results = []
            for child in active:
                returncode = child.process.wait(timeout=5)
                child.log_handle.close()
                final = read_final(child.metric_path)
                result = {
                    "round": round_number, "experiment": label, "gpu": child.gpu,
                    "seed": SEEDS[child.gpu], "run_id": child.run_id,
                    "returncode": returncode,
                    "status": "ok" if returncode == 0 and final else "failed",
                    "val_bpb": final.get("val_bpb") if final else None,
                    "metric_path": str(child.metric_path), "log_path": str(child.log_path),
                }
                state["runs"].append(result)
                round_results.append(result)
            write_state(state_path, state)
            post_event(control_event(
                args.campaign_id, "round_complete", round=round_number,
                experiment_label=label, results=round_results,
            ), dashboard)
            active = []
    finally:
        if active:
            terminate_children(active)
            for child in active:
                child.log_handle.close()

    state["status"] = "stopped" if stopping else "complete"
    state["finished_at"] = time.time()
    state.pop("active_runs", None)
    write_state(state_path, state)
    post_event(control_event(args.campaign_id, f"campaign_{state['status']}", runs=state["runs"]), dashboard)
    print(f"campaign {state['status']}: {state_path}", flush=True)
    return 130 if stopping else 0


if __name__ == "__main__":
    raise SystemExit(main())
