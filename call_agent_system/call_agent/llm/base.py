"""
Abstract LLM backend for the Call Agent.

Each concrete backend handles two tasks:
1. chat(system, history, user) -> produces the next operator utterance
2. extract(conversation)       -> produces a JSON object matching
   call_agent.config.REQUIRED_FIELDS

In simulated mode we do not call an LLM; instead we sample from the
profile in LLM_PROFILES to obtain a realistic (accuracy, latency) pair
and perturb the ground-truth record by the expected error rate.
"""

from __future__ import annotations

import json
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import LLM_PROFILES, REQUIRED_FIELDS


@dataclass
class LLMResult:
    text: str = ""
    json_data: Optional[Dict[str, Any]] = None
    latency: float = 0.0
    backend: str = ""
    tokens: int = 0


class BaseLLM(ABC):
    """Abstract LLM backend."""

    name: str = "base"

    _value_space = {
        "type":     ["fire", "accident", "medical", "crime", "other"],
        "severity": ["low", "medium", "high", "critical"],
        "emotion":  ["calm", "panic", "angry", "sad", "confused"],
        "services": ["police", "ambulance", "fire_truck", "rescue"],
        "hazards":  ["fire", "smoke", "gas", "weapon", "traffic", "water"],
    }

    def __init__(self, mode: str = "simulated", seed: int = 42,
                 temperature: float = 0.2):
        self.mode = mode
        self.temperature = temperature
        self._rng = random.Random(seed)
        self._profile = LLM_PROFILES.get(self.name, {})

    # --- chat --------------------------------------------------------------
    def chat(self, system: str, history: List[Dict[str, str]],
             user: str) -> LLMResult:
        t0 = time.perf_counter()
        if self.mode == "real":
            out = self._chat_real(system, history, user)
        else:
            out = self._chat_simulated(system, history, user)
        out.latency = max(out.latency, time.perf_counter() - t0)
        out.backend = self.name
        return out

    # --- extraction --------------------------------------------------------
    def extract(self, conversation: str,
                ground_truth: Optional[Dict[str, Any]] = None,
                noisy: bool = False,
                asr_wer: float = 0.0) -> LLMResult:
        t0 = time.perf_counter()
        if self.mode == "real":
            out = self._extract_real(conversation)
        else:
            out = self._extract_simulated(conversation, ground_truth,
                                          noisy, asr_wer)
        out.latency = max(out.latency, time.perf_counter() - t0)
        out.backend = self.name
        return out

    # --- real-mode hooks ---------------------------------------------------
    @abstractmethod
    def _chat_real(self, system: str, history: List[Dict[str, str]],
                   user: str) -> LLMResult: ...

    @abstractmethod
    def _extract_real(self, conversation: str) -> LLMResult: ...

    # --- simulated-mode defaults -------------------------------------------
    def _chat_simulated(self, system: str, history: List[Dict[str, str]],
                        user: str) -> LLMResult:
        latency = abs(self._rng.gauss(self._profile.get("latency", 2.0),
                                      self._profile.get("latency", 2.0) * 0.2))
        n_turns = sum(1 for m in history if m["role"] == "assistant")
        prompts = [
            "Emergency services, what is your emergency?",
            "What is your exact location?",
            "Are there any injuries or victims?",
            "Is there any fire or immediate hazard?",
            "How many people need assistance?",
            "Stay on the line, help is on the way.",
        ]
        reply = prompts[min(n_turns, len(prompts) - 1)]
        return LLMResult(text=reply, latency=latency, tokens=len(reply.split()))

    def _extract_simulated(self, conversation: str,
                           gt: Optional[Dict[str, Any]],
                           noisy: bool,
                           asr_wer: float = 0.0) -> LLMResult:
        """
        Effective accuracy depends on BOTH:
          * LLM's clean/noisy profile accuracy, AND
          * quality of upstream ASR transcript (asr_wer).
        Every WER pp above 5% costs a small fraction of accuracy.
        """
        acc = self._profile.get("extract_acc", 0.9)
        if noisy:
            acc = self._profile.get("robustness", 0.9)
        if asr_wer and asr_wer > 0.05:
            acc -= 0.45 * (asr_wer - 0.05)
        acc = max(0.4, acc)

        latency = abs(self._rng.gauss(self._profile.get("latency", 2.0),
                                      self._profile.get("latency", 2.0) * 0.2))
        if gt is None:
            return LLMResult(
                text="{}",
                json_data={f: None for f in REQUIRED_FIELDS},
                latency=latency)
        out: Dict[str, Any] = {}
        for field_name in REQUIRED_FIELDS:
            gt_val = gt.get(field_name)
            if self._rng.random() < acc:
                out[field_name] = gt_val
            else:
                out[field_name] = self._corrupt_field(field_name, gt_val)
        return LLMResult(text=json.dumps(out), json_data=out, latency=latency)

    def _corrupt_field(self, field_name: str, gt_val: Any) -> Any:
        space = self._value_space.get(field_name)
        if isinstance(gt_val, list):
            if space:
                return [self._rng.choice(space)]
            return []
        if space:
            wrong = [v for v in space if v != gt_val]
            return self._rng.choice(wrong) if wrong else gt_val
        if isinstance(gt_val, str) and gt_val.isdigit():
            delta = self._rng.choice([-1, 1, 2])
            return str(max(0, int(gt_val) + delta))
        if isinstance(gt_val, str):
            toks = gt_val.split()
            if len(toks) > 1:
                toks.pop(self._rng.randrange(len(toks)))
                return " ".join(toks)
        return gt_val

    @staticmethod
    def _parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
