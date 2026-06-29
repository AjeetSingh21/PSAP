"""Quick smoke test to verify the pipeline wiring."""

from __future__ import annotations

import sys
import os

# allow `python scripts/smoke_test.py` from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from call_agent.config import PipelineConfig
from call_agent.pipeline import CallAgent
from data.dataset import load_dataset


def main() -> int:
    calls = load_dataset(limit=3)
    print(f"Loaded {len(calls)} calls from outputs.json")

    cfg = PipelineConfig(asr="deepgram_nova2", llm="gemini_flash_2_5",
                         tts="deepgram_aura", mode="simulated")
    agent = CallAgent(cfg)
    for c in calls:
        t = agent.handle_call(c.clean_conversation, c.extracted_info,
                              call_id=c.call_id, noisy=c.is_noisy)
        print(f"\nCall #{c.call_id} | turns={t.turns} | "
              f"latency={t.total_latency:.2f}s\n  GT  : {c.extracted_info}"
              f"\n  Pred: {{k: v for k, v in t.extracted.items() if not k.startswith('_')}}")
        print("  Summary:", t.final_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
