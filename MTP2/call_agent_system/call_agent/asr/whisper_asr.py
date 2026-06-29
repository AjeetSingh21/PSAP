"""
OpenAI Whisper ASR backends (local inference).

Two flavours are exposed:

* :class:`WhisperLargeASR` - ``large-v3`` checkpoint, best accuracy.
* :class:`WhisperBaseASR`  - ``base`` checkpoint, lower latency on CPU.

For production both rely on the ``openai-whisper`` Python package, which
is imported lazily so the module can still be used in simulation mode on
machines without PyTorch.
"""

from __future__ import annotations

from .base import BaseASR, ASRResult


class _WhisperBase(BaseASR):
    """Shared real-mode plumbing for Whisper checkpoints."""

    checkpoint: str = "base"

    def _transcribe_real(self, audio_path: str) -> ASRResult:
        try:
            import whisper                                   # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "openai-whisper is required for mode='real'. "
                "Install with: pip install openai-whisper") from exc

        model = whisper.load_model(self.checkpoint)
        r = model.transcribe(audio_path, fp16=False)
        segments = r.get("segments", [])
        avg_conf = (sum(s.get("avg_logprob", -1) for s in segments) / len(segments)
                    if segments else -1.0)
        # convert log-prob (approx -1..0) to pseudo-confidence [0,1]
        confidence = float(max(0.0, min(1.0, (avg_conf + 1) if avg_conf else 0.8)))
        return ASRResult(text=r["text"].strip(), latency=0.0, confidence=confidence)


class WhisperLargeASR(_WhisperBase):
    """Whisper large-v3 (state-of-the-art open-source ASR)."""
    name = "whisper_large"
    checkpoint = "large-v3"


class WhisperBaseASR(_WhisperBase):
    """Whisper base (fast, CPU-friendly)."""
    name = "whisper_base"
    checkpoint = "base"
