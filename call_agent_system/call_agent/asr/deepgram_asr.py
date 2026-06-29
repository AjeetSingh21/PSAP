"""
Deepgram Nova-2 ASR backend.

Production path uses the official ``deepgram-sdk`` (optional import).
Simulated path uses the calibrated error model in :class:`BaseASR`.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import BaseASR, ASRResult


class DeepgramASR(BaseASR):
    """Deepgram Nova-2 streaming ASR."""

    name = "deepgram_nova2"

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key or os.environ.get("DEEPGRAM_API_KEY")

    def _transcribe_real(self, audio_path: str) -> ASRResult:
        """
        Call Deepgram's prerecorded API.  The SDK is imported lazily so
        the package can be imported in simulation-only environments.
        """
        try:
            from deepgram import DeepgramClient, PrerecordedOptions   # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "deepgram-sdk is required for mode='real'. "
                "Install with: pip install deepgram-sdk") from exc

        if not self.api_key:
            raise RuntimeError("DEEPGRAM_API_KEY environment variable is required.")

        dg = DeepgramClient(self.api_key)
        with open(audio_path, "rb") as fh:
            source = {"buffer": fh.read(), "mimetype": "audio/wav"}
        options = PrerecordedOptions(model="nova-2", language="en",
                                     smart_format=True, punctuate=True)
        response = dg.listen.rest.v("1").transcribe_file(source, options)
        alt = response.results.channels[0].alternatives[0]
        return ASRResult(text=alt.transcript, latency=0.0,
                         confidence=float(alt.confidence))
