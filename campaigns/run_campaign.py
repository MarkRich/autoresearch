#!/usr/bin/env python3
"""Run a finite, independent-lane two-GPU autoresearch campaign.

The default plan sends the same treatment to both GPU lanes with different
fixed seeds. Explicit JSON plans can instead run crossover experiments while
preserving one process per GPU and complete provenance for every lane.
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
DEFAULT_MAX_GPU_TEMP_C = 88
SEEDS = (101, 202)

BASE_ENV = {
    "DEVICE_BATCH_SIZE": "8",
    "TOTAL_BATCH_SIZE": str(2**19),
    "AMP_DTYPE": "bf16",
    "OPENCLAW_FORCE_SDPA": "1",
    "OPENCLAW_DISABLE_TORCH_COMPILE": "0",
    "UNCOUNTED_WARMUP_STEPS": "11",
    "WINDOW_PATTERN": "LLLL",
    "DEPTH": "8",
    "ASPECT_RATIO": "64",
    "HEAD_DIM": "128",
    "USE_VALUE_EMBEDS": "0",
    "MLP_KIND": "relu_squared",
    "ATTN_RESIDUAL_MODE": "none",
    "ATTN_RESIDUAL_BACKEND": "package",
    "ATTN_RESIDUAL_BLOCK_SIZE": "4",
    "ATTN_OUTPUT_GATE": "none",
    "OPTIMIZER_KIND": "muon",
    "FP32_ADAM_STATE": "0",
    "EMBEDDING_LR": "0.0025",
    "UNEMBEDDING_LR": "0.0005",
    "MATRIX_LR": "0.0005",
    "SCALAR_LR": "0.0025",
    "WEIGHT_DECAY": "0.01",
    "ADAM_BETA1": "0.8",
    "ADAM_BETA2": "0.95",
    "WARMUP_RATIO": "0.15",
    "WARMDOWN_RATIO": "0.60",
    "FINAL_LR_FRAC": "0.05",
    "TRAIN_PROBE_BATCHES": "4",
    "GRAD_CLIP_NORM": "0",
    # Random-loss checks are disabled for these short, scheduled runs, but a
    # sustained fixed-probe regression still saves GPU time on clear collapse.
    "FAILFAST_MIN_PROGRESS": "2.0",
    "FAILFAST_REGRESSION_MIN_RISE": "0.50",
    "FAILFAST_REGRESSION_PATIENCE_EVENTS": "3",
}

PLAN = (
    ("control", {}),
    ("value_residuals", {"USE_VALUE_EMBEDS": "1"}),
    ("matrix_lr_eighth", {"MATRIX_LR": "0.00025"}),
    ("scalar_lr_eighth", {"SCALAR_LR": "0.00125"}),
    ("warmdown_70", {"WARMDOWN_RATIO": "0.70"}),
    ("zero_final_lr", {"FINAL_LR_FRAC": "0.0"}),
)

ALLOWED_OVERRIDE_KEYS = frozenset(BASE_ENV)


def resolved_lane_env(overrides: dict[str, str]) -> dict[str, str]:
    """Return the exact non-secret training environment for a lane."""
    return {**BASE_ENV, **overrides}


@dataclass(frozen=True)
class LaneSpec:
    gpu: int
    seed: int
    label: str
    overrides: dict[str, str]


@dataclass
class ChildRun:
    gpu: int
    seed: int
    label: str
    overrides: dict[str, str]
    run_id: str
    log_path: Path
    metric_path: Path
    process: subprocess.Popen[Any]
    log_handle: Any


def paired_plan() -> list[list[LaneSpec]]:
    return [
        [
            LaneSpec(gpu=gpu, seed=SEEDS[gpu], label=label, overrides=dict(overrides))
            for gpu in range(len(SEEDS))
        ]
        for label, overrides in PLAN
    ]


def load_plan(path: Path) -> list[list[LaneSpec]]:
    """Load a two-lane experiment plan with explicit, allow-listed overrides."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_rounds = payload.get("rounds")
    if not isinstance(raw_rounds, list) or not raw_rounds:
        raise ValueError("plan must contain a non-empty 'rounds' list")
    rounds: list[list[LaneSpec]] = []
    for round_number, raw_round in enumerate(raw_rounds, start=1):
        raw_lanes = raw_round.get("lanes") if isinstance(raw_round, dict) else None
        if not isinstance(raw_lanes, list) or len(raw_lanes) != len(SEEDS):
            raise ValueError(f"round {round_number} must define exactly {len(SEEDS)} lanes")
        lanes: list[LaneSpec] = []
        for raw_lane in raw_lanes:
            if not isinstance(raw_lane, dict):
                raise ValueError(f"round {round_number} contains a non-object lane")
            gpu = raw_lane.get("gpu")
            label = raw_lane.get("label")
            overrides = raw_lane.get("env", {})
            if gpu not in range(len(SEEDS)) or not isinstance(label, str) or not label:
                raise ValueError(f"round {round_number} has an invalid gpu or label")
            if not isinstance(overrides, dict):
                raise ValueError(f"round {round_number} lane {gpu} env must be an object")
            unknown = set(overrides) - ALLOWED_OVERRIDE_KEYS
            if unknown:
                raise ValueError(f"round {round_number} lane {gpu} has unsupported env keys: {sorted(unknown)}")
            lanes.append(LaneSpec(
                gpu=gpu,
                seed=SEEDS[gpu],
                label=label,
                overrides={key: str(value) for key, value in overrides.items()},
            ))
        if sorted(lane.gpu for lane in lanes) != list(range(len(SEEDS))):
            raise ValueError(f"round {round_number} must assign each GPU exactly once")
        rounds.append(sorted(lanes, key=lambda lane: lane.gpu))
    return rounds


def serialize_plan(plan: list[list[LaneSpec]]) -> list[dict[str, Any]]:
    return [
        {
            "round": round_number,
            "lanes": [
                {
                    "gpu": lane.gpu,
                    "seed": lane.seed,
                    "label": lane.label,
                    "env": dict(lane.overrides),
                }
                for lane in lanes
            ],
        }
        for round_number, lanes in enumerate(plan, start=1)
    ]


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


def parse_gpu_temperature(output: str) -> int:
    value = int(float(output.strip().splitlines()[0]))
    if not 0 <= value <= 120:
        raise ValueError(f"implausible GPU temperature: {value}")
    return value


def read_gpu_temperature(gpu: int) -> int | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi", f"--id={gpu}",
                "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
        return parse_gpu_temperature(output)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


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


def launch_round(
    campaign_id: str,
    round_number: int,
    lanes: list[LaneSpec],
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
    work_tag = os.environ.get("CODEX_WORK_TAG", f"autoresearch:{campaign_id}")
    for lane in lanes:
        gpu, seed, label, overrides = lane.gpu, lane.seed, lane.label, lane.overrides
        run_id = f"{campaign_id}-r{round_number:02d}-gpu{gpu}-{label}-s{seed}"
        log_path = logs / f"{run_id}.log"
        metric_path = metrics / f"{run_id}.jsonl"
        env = os.environ.copy()
        env.update(resolved_lane_env(overrides))
        env.update(dashboard)
        env.update({
            "CODEX_WORK_TAG": work_tag,
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
        children.append(ChildRun(
            gpu, seed, label, dict(overrides), run_id,
            log_path, metric_path, process, log_handle,
        ))
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
    parser.add_argument("--plan-file", type=Path)
    parser.add_argument("--campaign-id", default=f"codex-30m-{utc_stamp().lower()}")
    parser.add_argument("--python", default="/home/mark/autoresearch-transformer-paper/.venv/bin/python")
    parser.add_argument("--dashboard-env", type=Path, default=Path.home() / ".config/autoresearch-dashboard.env")
    parser.add_argument("--max-gpu-temp-c", type=int, default=DEFAULT_MAX_GPU_TEMP_C)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.budget_seconds < 30:
        raise SystemExit("budget must be at least 30 seconds")
    if not 70 <= args.max_gpu_temp_c <= 95:
        raise SystemExit("max GPU temperature must be between 70C and 95C")
    full_plan = load_plan(args.plan_file) if args.plan_file else paired_plan()
    round_plan = full_plan[: max(1, min(args.rounds, len(full_plan)))]
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
        "max_gpu_temp_c": args.max_gpu_temp_c,
        "resolved_plan": serialize_plan(round_plan),
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
        for round_number, lanes in enumerate(round_plan, start=1):
            if stopping:
                break
            labels = [lane.label for lane in lanes]
            post_event(control_event(
                args.campaign_id, "round_start", round=round_number,
                experiment_labels=labels,
            ), dashboard)
            active = launch_round(
                args.campaign_id, round_number, lanes, args.budget_seconds,
                args.python, dashboard, code_sha, campaign_root,
            )
            hard_deadline = time.time() + args.budget_seconds + 600
            while not stopping and any(child.process.poll() is None for child in active):
                if time.time() > hard_deadline:
                    print(f"round {round_number} exceeded hard deadline; terminating", flush=True)
                    terminate_children(active)
                    break
                temperatures = {
                    str(child.gpu): read_gpu_temperature(child.gpu) for child in active
                }
                state["gpu_temperatures_c"] = temperatures
                too_hot = {
                    gpu: temp for gpu, temp in temperatures.items()
                    if temp is not None and temp >= args.max_gpu_temp_c
                }
                if too_hot:
                    state["stop_reason"] = "thermal_limit"
                    state["thermal_limit_readings_c"] = too_hot
                    print(
                        f"thermal limit reached ({too_hot}); stopping campaign safely",
                        flush=True,
                    )
                    post_event(control_event(
                        args.campaign_id, "thermal_stop",
                        temperatures_c=temperatures,
                        max_gpu_temp_c=args.max_gpu_temp_c,
                    ), dashboard)
                    stopping = True
                    terminate_children(active)
                    break
                state["active_round"] = round_number
                state["active_experiments"] = labels
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
                    "round": round_number, "experiment": child.label, "gpu": child.gpu,
                    "seed": child.seed,
                    "env": resolved_lane_env(child.overrides),
                    "env_overrides": child.overrides,
                    "run_id": child.run_id,
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
                experiment_labels=labels, results=round_results,
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
