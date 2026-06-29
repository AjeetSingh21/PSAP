"""
Batch evaluation runner.

Iterates over the cartesian product of all registered ASR x LLM x TTS
backends, runs every call from the benchmark once per combination, plus
two extra seeds per combination (for the Consistency metric), and
aggregates the results via :mod:`evaluation.metrics`.
"""

from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from call_agent.config import PipelineConfig, ASR_PROFILES, LLM_PROFILES, TTS_PROFILES
from call_agent.pipeline import CallAgent
from data.dataset import Call, load_dataset

from .metrics import CombinationResult, evaluate_predictions


def _run_once(cfg: PipelineConfig, calls: Sequence[Call]) -> List:
    """Run the pipeline over all calls with a single config / seed."""
    agent = CallAgent(cfg)
    traces = []
    for c in calls:
        trace = agent.handle_call(
            clean_conversation=c.clean_conversation,
            ground_truth=c.extracted_info,
            call_id=c.call_id,
            noisy=c.is_noisy,
        )
        traces.append(trace)
    return traces


def run_single_combination(asr: str, llm: str, tts: str,
                           calls: Sequence[Call],
                           seeds: Sequence[int] = (42, 101, 202),
                           ) -> CombinationResult:
    """Run one combination across multiple seeds and return aggregated result."""
    primary_traces = _run_once(
        PipelineConfig(asr=asr, llm=llm, tts=tts,
                       mode="simulated", seed=seeds[0]),
        calls)
    extra_preds = []
    for s in seeds[1:]:
        extra_traces = _run_once(
            PipelineConfig(asr=asr, llm=llm, tts=tts,
                           mode="simulated", seed=s),
            calls)
        extra_preds.append([t.extracted for t in extra_traces])

    return evaluate_predictions(
        traces=primary_traces,
        ground_truth=[c.extracted_info for c in calls],
        is_noisy=[c.is_noisy for c in calls],
        per_seed_predictions=extra_preds,
        asr=asr, llm=llm, tts=tts,
    )


def run_all_combinations(dataset_path: Optional[str] = None,
                         limit: Optional[int] = None,
                         out_dir: str = "results",
                         seeds: Sequence[int] = (42, 101, 202),
                         ) -> List[CombinationResult]:
    """Run every (asr, llm, tts) combination and write ``combo_results.json``."""
    calls = load_dataset(dataset_path, limit=limit)
    combos = list(itertools.product(ASR_PROFILES, LLM_PROFILES, TTS_PROFILES))
    total = len(combos)
    results: List[CombinationResult] = []

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    for i, (asr, llm, tts) in enumerate(combos, 1):
        ts = time.perf_counter()
        res = run_single_combination(asr, llm, tts, calls, seeds=seeds)
        results.append(res)
        dt = time.perf_counter() - ts
        print(f"[{i:3}/{total}] {asr:<16} {llm:<18} {tts:<14} "
              f"TSR={res.task_success_rate:.3f} "
              f"Acc={res.accuracy_overall:.3f} "
              f"CallLat={res.call_latency_mean:.2f}s "
              f"Rob={res.robustness:.3f} Cons={res.consistency:.3f} "
              f"({dt:.1f}s)")

    # Save aggregated JSON -----------------------------------------------------
    Path(out_dir, "combo_results.json").write_text(
        json.dumps([r.as_dict() for r in results], indent=2))
    print(f"\nAll combinations done in {time.perf_counter()-t_start:.1f}s.")
    return results
