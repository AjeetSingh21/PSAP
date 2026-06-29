"""
End-to-end evaluation script.

Runs every (ASR x LLM x TTS) combination over ``--limit`` calls of the
benchmark and writes the aggregated metrics JSON plus a comparison CSV.

Usage::

    python evaluate.py --limit 200 --out results
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from evaluation.runner import run_all_combinations


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Full evaluation harness.")
    p.add_argument("--dataset", default=None,
                   help="Path to outputs.json (defaults to uploads/).")
    p.add_argument("--limit", type=int, default=200,
                   help="Number of calls per combination.")
    p.add_argument("--out", default="results",
                   help="Output directory for JSON / CSV.")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202],
                   help="Seeds (first is the primary run; extras are for consistency).")
    args = p.parse_args(argv)

    results = run_all_combinations(
        dataset_path=args.dataset,
        limit=args.limit,
        out_dir=args.out,
        seeds=tuple(args.seeds),
    )

    # --- Write a wide CSV for easy paper-style tables -------------------
    out_dir = Path(args.out)
    fields = ["asr", "llm", "tts", "n_calls",
              "task_success_rate", "accuracy_overall",
              "call_latency_mean", "call_latency_p95",
              "avg_response_time", "robustness", "consistency",
              "asr_latency_mean", "llm_latency_mean", "tts_latency_mean",
              "accuracy_clean", "accuracy_noisy"]
    with open(out_dir / "combo_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: getattr(r, k) for k in fields}
            w.writerow(row)

    # --- Per-field accuracy CSV -----------------------------------------
    all_fields = sorted({f for r in results for f in r.accuracy_by_field})
    with open(out_dir / "field_accuracy.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["asr", "llm", "tts"] + all_fields)
        for r in results:
            w.writerow([r.asr, r.llm, r.tts] +
                       [r.accuracy_by_field.get(f, 0.0) for f in all_fields])
    print("Wrote:", out_dir / "combo_results.csv")
    print("Wrote:", out_dir / "field_accuracy.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
