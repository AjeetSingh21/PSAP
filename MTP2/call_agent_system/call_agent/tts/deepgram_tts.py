"""Deepgram Aura TTS backend (streaming, human-sounding)."""

from __future__ import annotations

import os
from typing import Optional

from .base import BaseTTS, TTSResult


class DeepgramAuraTTS(BaseTTS):
    name = "deepgram_aura"

    def _speak_real(self, text: str, out_path: Optional[str]) -> TTSResult:
        try:
            from deepgram import DeepgramClient, SpeakOptions   # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "deepgram-sdk required for mode='real'. "
                "pip install deepgram-sdk") from exc
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPGRAM_API_KEY not set.")
        out = out_path or "/tmp/aura.wav"
        dg = DeepgramClient(api_key)
        dg.speak.rest.v("1").save(out, {"text": text},
                                  SpeakOptions(model="aura-asteria-en"))
        return TTSResult(audio_path=out, latency=0.0)
