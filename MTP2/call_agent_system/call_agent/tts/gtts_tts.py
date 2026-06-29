"""gTTS (Google translate backend, free and easy to install)."""

from __future__ import annotations

from typing import Optional

from .base import BaseTTS, TTSResult


class GTTSTTS(BaseTTS):
    name = "gtts"

    def _speak_real(self, text: str, out_path: Optional[str]) -> TTSResult:
        try:
            from gtts import gTTS                             # type: ignore
        except ImportError as exc:
            raise RuntimeError("gtts required. pip install gTTS") from exc
        out = out_path or "/tmp/gtts.mp3"
        gTTS(text=text, lang="en", tld="co.in").save(out)
        return TTSResult(audio_path=out, latency=0.0)
