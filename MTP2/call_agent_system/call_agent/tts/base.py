"""
Abstract Text-to-Speech backend.

The contract is extremely thin: :meth:`BaseTTS.speak` takes a text
string and returns a :class:`TTSResult` with the path of the generated
audio file (real mode) or a dummy path (simulated mode) plus the
synthesis latency, which is all the evaluation harness needs.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..config import TTS_PROFILES


@dataclass
class TTSResult:
    audio_path: Optional[str]
    latency: float
    mos: float = 0.0
    backend: str = ""


class BaseTTS(ABC):
    name: str = "base"

    def __init__(self, mode: str = "simulated", seed: int = 42):
        self.mode = mode
        self._rng = random.Random(seed)
        self._profile = TTS_PROFILES.get(self.name, {})

    def speak(self, text: str, out_path: Optional[str] = None) -> TTSResult:
        t0 = time.perf_counter()
        if self.mode == "real":
            result = self._speak_real(text, out_path)
        else:
            result = self._speak_simulated(text)
        result.latency = max(result.latency, time.perf_counter() - t0)
        result.backend = self.name
        result.mos = self._profile.get("mos", 3.5)
        return result

    @abstractmethod
    def _speak_real(self, text: str,
                    out_path: Optional[str]) -> TTSResult: ...

    def _speak_simulated(self, text: str) -> TTSResult:
        # Latency grows with text length to approximate real streaming TTS.
        base = self._profile.get("latency", 0.5)
        n_chars = max(1, len(text))
        latency = abs(self._rng.gauss(base + n_chars * 0.002, base * 0.1))
        return TTSResult(audio_path=None, latency=latency)
