"""
Abstract base class for every Automatic Speech Recognition backend.

A concrete backend is only required to implement
:meth:`BaseASR._transcribe_real` (for production) and
:meth:`BaseASR._transcribe_simulated` (used by the evaluation harness).

The public :meth:`BaseASR.transcribe` method chooses between the two based
on :attr:`BaseASR.mode`, measures wall-clock latency, and returns a
uniform :class:`ASRResult` object.
"""

from __future__ import annotations

import random
import time
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..config import ASR_PROFILES


@dataclass
class ASRResult:
    """Output contract that every ASR backend must satisfy."""
    text: str
    latency: float      # seconds, end-to-end
    confidence: float   # [0, 1]
    wer_estimated: float = 0.0   # only populated in simulated mode
    backend: str = ""


class BaseASR(ABC):
    """Abstract ASR backend."""

    #: unique identifier used by the factory and config
    name: str = "base"

    def __init__(self, mode: str = "simulated", seed: int = 42):
        self.mode = mode
        self._rng = random.Random(seed)
        self._profile = ASR_PROFILES.get(self.name, {})

    # ---- public API --------------------------------------------------------
    def transcribe(self, audio: str, *, noisy: bool = False,
                   clean_reference: Optional[str] = None) -> ASRResult:
        """
        Transcribe an ``audio`` payload.

        ``audio`` is treated as a file path in ``mode='real'`` and as the
        clean reference text in ``mode='simulated'``.

        The ``noisy`` flag toggles between the clean and noisy WER profile.
        ``clean_reference`` is used only by the simulator to introduce
        realistic character-level errors.
        """
        t0 = time.perf_counter()
        if self.mode == "real":
            result = self._transcribe_real(audio)
        else:
            ref = clean_reference if clean_reference is not None else audio
            result = self._transcribe_simulated(ref, noisy=noisy)
        result.latency = max(result.latency, time.perf_counter() - t0)
        result.backend = self.name
        return result

    # ---- hooks -------------------------------------------------------------
    @abstractmethod
    def _transcribe_real(self, audio_path: str) -> ASRResult:
        """Call the real cloud/local ASR API.  Override in subclass."""

    def _transcribe_simulated(self, reference: str, *, noisy: bool) -> ASRResult:
        """
        Default simulator: corrupts the reference text at a per-character
        rate drawn from the backend's WER profile.  Subclasses can override
        this for backend-specific quirks (e.g. Whisper tends to drop
        function words, Deepgram tends to substitute numbers).
        """
        target_wer = self._profile.get("wer_noisy" if noisy else "wer_mean", 0.1)
        latency = abs(self._rng.gauss(self._profile.get("latency", 1.0),
                                      self._profile.get("latency", 1.0) * 0.15))
        text = self._corrupt(reference, target_wer)
        confidence = max(0.55, min(0.99,
                                   self._profile.get("confidence_bias", 0.85)
                                   + self._rng.gauss(0, 0.03)))
        return ASRResult(text=text, latency=latency,
                         confidence=confidence, wer_estimated=target_wer)

    # ---- helpers -----------------------------------------------------------
    def _corrupt(self, text: str, wer: float) -> str:
        """Introduce substitution / deletion / insertion errors at ~WER rate."""
        if not text:
            return text
        chars = list(text)
        n_errors = max(0, int(len(chars) * wer * 0.6))  # char-level approx
        for _ in range(n_errors):
            idx = self._rng.randrange(len(chars))
            op = self._rng.choice(("sub", "sub", "sub", "del", "ins"))
            if op == "sub":
                chars[idx] = self._rng.choice(string.ascii_lowercase)
            elif op == "del" and len(chars) > 1:
                chars.pop(idx)
            else:  # ins
                chars.insert(idx, self._rng.choice(string.ascii_lowercase))
        return "".join(chars)
