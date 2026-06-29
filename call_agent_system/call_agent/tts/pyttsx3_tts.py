"""Offline pyttsx3 backend (espeak/SAPI wrapper)."""

from __future__ import annotations

from typing import Optional

from .base import BaseTTS, TTSResult


class Pyttsx3TTS(BaseTTS):
    name = "pyttsx3"

    def _speak_real(self, text: str, out_path: Optional[str]) -> TTSResult:
        try:
            import pyttsx3                                    # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyttsx3 required. pip install pyttsx3") from exc
        engine = pyttsx3.init()
        out = out_path or "/tmp/pyttsx3.wav"
        engine.save_to_file(text, out)
        engine.runAndWait()
        return TTSResult(audio_path=out, latency=0.0)
