"""
Minimal CLI entry point: run ONE call through a chosen (ASR, LLM, TTS)
combination and print the Resource-Allocation handoff.

Usage::

    python main.py --asr whisper_large --llm gemini_flash_2_5 \
                   --tts deepgram_aura --call-id 5
"""

from __future__ import annotations

import argparse
import json
import sys

from call_agent.config import PipelineConfig
from call_agent.pipeline import CallAgent
from data.dataset import load_dataset


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Run a single call through the Call Agent.")
    p.add_argument("--asr", default="deepgram_nova2")
    p.add_argument("--llm", default="gemini_flash_2_5")
    p.add_argument("--tts", default="deepgram_aura")
    p.add_argument("--call-id", type=int, default=1)
    p.add_argument("--dataset", default=None)
    p.add_argument("--mode", default="simulated", choices=("simulated", "real"))
    args = p.parse_args(argv)

    calls = load_dataset(args.dataset)
    target = next((c for c in calls if c.call_id == args.call_id), None)
    if target is None:
        print(f"call_id {args.call_id} not found.", file=sys.stderr)
        return 1

    cfg = PipelineConfig(asr=args.asr, llm=args.llm, tts=args.tts, mode=args.mode)
    agent = CallAgent(cfg)
    trace = agent.handle_call(
        clean_conversation=target.clean_conversation,
        ground_truth=target.extracted_info,
        call_id=target.call_id,
        noisy=target.is_noisy,
    )

    print(json.dumps({
        "call_id":       trace.call_id,
        "backends":      trace.backends,
        "turns":         trace.turns,
        "total_latency": round(trace.total_latency, 3),
        "extracted":     {k: v for k, v in trace.extracted.items()
                          if not k.startswith("_")},
        "summary":       trace.final_summary,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
