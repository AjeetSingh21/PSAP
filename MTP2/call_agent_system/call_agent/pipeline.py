"""
CallAgent - orchestrates the ASR -> LLM -> TTS loop (Figure 3 of the MTP report).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .config import PipelineConfig
from .asr import get_asr
from .llm import get_llm
from .tts import get_tts
from .extractor import InformationExtractor
from .prompts import DISPATCHER_SYSTEM_PROMPT


@dataclass
class CallTrace:
    call_id: Any = None
    asr_latencies: List[float] = field(default_factory=list)
    llm_latencies: List[float] = field(default_factory=list)
    tts_latencies: List[float] = field(default_factory=list)
    total_latency: float = 0.0
    turns: int = 0
    transcript: str = ""
    operator_utterances: List[str] = field(default_factory=list)
    extracted: Dict[str, Any] = field(default_factory=dict)
    final_summary: str = ""
    backends: Dict[str, str] = field(default_factory=dict)


class CallAgent:
    """End-to-end Call Agent combining an ASR, an LLM and a TTS backend."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.asr = get_asr(cfg.asr, mode=cfg.mode, seed=cfg.seed)
        self.llm = get_llm(cfg.llm, mode=cfg.mode, seed=cfg.seed)
        self.tts = get_tts(cfg.tts, mode=cfg.mode, seed=cfg.seed)
        self.extractor = InformationExtractor(self.llm)

    def handle_call(self,
                    clean_conversation: str,
                    ground_truth: Optional[Dict[str, Any]] = None,
                    call_id: Any = None,
                    noisy: bool = False) -> CallTrace:
        trace = CallTrace(call_id=call_id)
        trace.backends = {"asr": self.cfg.asr, "llm": self.cfg.llm,
                          "tts": self.cfg.tts}

        t0 = time.perf_counter()
        history: List[Dict[str, str]] = []

        caller_turns = [line.split(":", 1)[1].strip()
                        for line in clean_conversation.split("\n")
                        if line.startswith("Caller:")]

        full_transcript_lines: List[str] = []
        for caller_utt in caller_turns[: self.cfg.max_turns]:
            asr_out = self.asr.transcribe(caller_utt, noisy=noisy,
                                          clean_reference=caller_utt)
            trace.asr_latencies.append(asr_out.latency)
            transcribed = asr_out.text
            full_transcript_lines.append("Caller: " + transcribed)
            history.append({"role": "user", "content": transcribed})

            llm_out = self.llm.chat(DISPATCHER_SYSTEM_PROMPT, history,
                                    transcribed)
            trace.llm_latencies.append(llm_out.latency)
            operator_reply = llm_out.text
            trace.operator_utterances.append(operator_reply)
            history.append({"role": "assistant", "content": operator_reply})
            full_transcript_lines.append("Operator: " + operator_reply)

            tts_out = self.tts.speak(operator_reply)
            trace.tts_latencies.append(tts_out.latency)

            trace.turns += 1

        trace.transcript = "\n".join(full_transcript_lines)

        # Couple extraction accuracy to upstream ASR quality
        if hasattr(self.asr, "_profile"):
            asr_wer = self.asr._profile.get(
                "wer_noisy" if noisy else "wer_mean", 0.0)
        else:
            asr_wer = 0.0

        extracted = self.extractor.extract(trace.transcript,
                                           ground_truth=ground_truth,
                                           noisy=noisy,
                                           asr_wer=asr_wer)
        extract_latency = float(extracted.pop("_latency", 0.0))
        _ = extracted.pop("_backend", None)
        trace.extracted = extracted
        trace.final_summary = self._summarise(extracted)

        wall_clock = time.perf_counter() - t0
        simulated_total = (sum(trace.asr_latencies) + sum(trace.llm_latencies)
                           + sum(trace.tts_latencies) + extract_latency)
        trace.total_latency = max(wall_clock, simulated_total)
        return trace

    @staticmethod
    def _summarise(extracted: Dict[str, Any]) -> str:
        sev = str(extracted.get("severity", "?")).title()
        typ = extracted.get("type", "?")
        loc = extracted.get("location", "unknown location")
        vic = extracted.get("victims", "?")
        svc = extracted.get("services") or ["TBD"]
        return (sev + " " + str(typ) + " at " + str(loc) + ", "
                + str(vic) + " victim(s); dispatch " + ", ".join(svc) + ".")
