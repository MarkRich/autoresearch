#!/usr/bin/env python3
"""Print a compact, reproducible summary from a campaign state file."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run.get("status") == "ok" and run.get("val_bpb") is not None:
            grouped[str(run["experiment"])].append(run)
    rows = []
    for experiment, items in grouped.items():
        scores = [float(item["val_bpb"]) for item in items]
        rows.append({
            "experiment": experiment,
            "n": len(scores),
            "mean_val_bpb": statistics.fmean(scores),
            "min_val_bpb": min(scores),
            "max_val_bpb": max(scores),
            "spread": max(scores) - min(scores),
            "scores": scores,
        })
    return sorted(rows, key=lambda row: (row["mean_val_bpb"], row["experiment"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", type=Path)
    args = parser.parse_args()
    state = json.loads(args.state.read_text(encoding="utf-8"))
    rows = summarize_runs(state.get("runs", []))
    print("experiment\tn\tmean_val_bpb\tspread\tscores")
    for row in rows:
        scores = ",".join(f"{score:.6f}" for score in row["scores"])
        print(
            f"{row['experiment']}\t{row['n']}\t{row['mean_val_bpb']:.6f}\t"
            f"{row['spread']:.6f}\t{scores}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
