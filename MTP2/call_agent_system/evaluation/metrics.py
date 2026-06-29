"""
Evaluation metrics for the end-to-end Call Agent.

All six metrics requested for MTP-2 are implemented here:

1. **Task Success Rate (TSR)**
   Fraction of calls where every CRITICAL field (type, severity, location,
   services) is correctly extracted.

2. **Accuracy**
   Field-level average across every REQUIRED field (micro-averaged).

3. **Call Latency**
   Mean total wall-clock time per call (ASR + LLM + TTS loop, all turns).

4. **Average Response Time**
   Mean of (ASR + LLM + TTS) latency per conversational turn, averaged
   across all turns of all calls.  This is the latency the caller
   experiences between speaking and hearing the operator reply.

5. **Robustness**
   Relative accuracy on noisy calls / accuracy on clean calls.
   A robustness of 1.0 means the backend does not degrade under noise.

6. **Consistency**
   1 - (mean of per-call TSR standard deviation across independent seeds).
   Values close to 1.0 mean the backend returns the same answer for the
   same call every time.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Sequence

from call_agent.config import REQUIRED_FIELDS, CRITICAL_FIELDS


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class CombinationResult:
    asr: str
    llm: str
    tts: str
    n_calls: int = 0

    # Primary metrics --------------------------------------------------------
    task_success_rate: float = 0.0
    accuracy_overall: float = 0.0
    accuracy_by_field: Dict[str, float] = field(default_factory=dict)
    call_latency_mean: float = 0.0
    call_latency_p95: float = 0.0
    avg_response_time: float = 0.0
    robustness: float = 0.0
    consistency: float = 0.0

    # Diagnostics ------------------------------------------------------------
    asr_latency_mean: float = 0.0
    llm_latency_mean: float = 0.0
    tts_latency_mean: float = 0.0
    accuracy_clean: float = 0.0
    accuracy_noisy: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Per-call scoring helpers
# ---------------------------------------------------------------------------
def _eq(a: Any, b: Any) -> bool:
    """List-aware equality used for field comparison."""
    if isinstance(a, list) or isinstance(b, list):
        la = a if isinstance(a, list) else ([a] if a else [])
        lb = b if isinstance(b, list) else ([b] if b else [])
        return set(map(str, la)) == set(map(str, lb))
    if a is None or b is None:
        return a == b
    return str(a).strip().lower() == str(b).strip().lower()


def score_call(predicted: Dict[str, Any],
               ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Return per-field correctness plus aggregate flags."""
    per_field = {f: _eq(predicted.get(f), ground_truth.get(f))
                 for f in REQUIRED_FIELDS}
    critical_ok = all(per_field[f] for f in CRITICAL_FIELDS)
    overall_acc = sum(per_field.values()) / len(per_field)
    return {"per_field": per_field,
            "critical_ok": critical_ok,
            "overall_acc": overall_acc}


# ---------------------------------------------------------------------------
# Aggregation over one (asr, llm, tts) combination
# ---------------------------------------------------------------------------
def evaluate_predictions(traces: Sequence[Any],
                         ground_truth: Sequence[Dict[str, Any]],
                         is_noisy: Sequence[bool],
                         per_seed_predictions: Sequence[Sequence[Dict[str, Any]]] = (),
                         asr: str = "",
                         llm: str = "",
                         tts: str = "") -> CombinationResult:
    """
    Compute all 6 metrics for one (asr, llm, tts) combination.

    Parameters
    ----------
    traces : sequence of CallTrace
        One per call in the evaluation set.
    ground_truth : sequence of dict
        Ground-truth records aligned with ``traces``.
    is_noisy : sequence of bool
        Which calls belong to the noisy bucket (for robustness).
    per_seed_predictions :
        Optional extra predictions (one list per additional seed) used
        *only* for consistency.  Each inner list has the same length as
        ``traces``.
    """
    n = len(traces)
    per_field_totals: Dict[str, List[bool]] = {f: [] for f in REQUIRED_FIELDS}
    critical_flags: List[bool] = []
    overall_accs: List[float] = []
    call_latencies: List[float] = []
    response_times: List[float] = []
    asr_latencies: List[float] = []
    llm_latencies: List[float] = []
    tts_latencies: List[float] = []
    clean_overall: List[float] = []
    noisy_overall: List[float] = []

    for idx, (trace, gt, noisy) in enumerate(zip(traces, ground_truth, is_noisy)):
        pred = trace.extracted
        sc = score_call(pred, gt)
        for f, ok in sc["per_field"].items():
            per_field_totals[f].append(ok)
        critical_flags.append(sc["critical_ok"])
        overall_accs.append(sc["overall_acc"])
        (noisy_overall if noisy else clean_overall).append(sc["overall_acc"])
        call_latencies.append(trace.total_latency)
        if trace.turns:
            turn_rt = [(a + l + t) for a, l, t in zip(
                trace.asr_latencies, trace.llm_latencies, trace.tts_latencies)]
            response_times.extend(turn_rt)
            asr_latencies.extend(trace.asr_latencies)
            llm_latencies.extend(trace.llm_latencies)
            tts_latencies.extend(trace.tts_latencies)

    # Consistency --------------------------------------------------------------
    consistency = 1.0
    if per_seed_predictions:
        per_call_stds = []
        for i in range(n):
            crit_per_seed = []
            for seed_preds in per_seed_predictions:
                sc = score_call(seed_preds[i], ground_truth[i])
                crit_per_seed.append(int(sc["critical_ok"]))
            if len(crit_per_seed) > 1:
                per_call_stds.append(statistics.pstdev(crit_per_seed))
        if per_call_stds:
            consistency = 1.0 - statistics.mean(per_call_stds)

    # Robustness ---------------------------------------------------------------
    acc_clean = (sum(clean_overall) / len(clean_overall)) if clean_overall else 0.0
    acc_noisy = (sum(noisy_overall) / len(noisy_overall)) if noisy_overall else acc_clean
    robustness = (acc_noisy / acc_clean) if acc_clean else 1.0
    robustness = min(max(robustness, 0.0), 1.0)

    p95 = (sorted(call_latencies)[int(0.95 * len(call_latencies))]
           if call_latencies else 0.0)

    return CombinationResult(
        asr=asr, llm=llm, tts=tts, n_calls=n,
        task_success_rate=sum(critical_flags) / n if n else 0.0,
        accuracy_overall=sum(overall_accs) / n if n else 0.0,
        accuracy_by_field={f: sum(v) / len(v) if v else 0.0
                           for f, v in per_field_totals.items()},
        call_latency_mean=statistics.mean(call_latencies) if call_latencies else 0.0,
        call_latency_p95=p95,
        avg_response_time=statistics.mean(response_times) if response_times else 0.0,
        robustness=robustness,
        consistency=consistency,
        asr_latency_mean=statistics.mean(asr_latencies) if asr_latencies else 0.0,
        llm_latency_mean=statistics.mean(llm_latencies) if llm_latencies else 0.0,
        tts_latency_mean=statistics.mean(tts_latencies) if tts_latencies else 0.0,
        accuracy_clean=acc_clean, accuracy_noisy=acc_noisy,
    )
