"""
Gemma-3 7B backend (local, served through Ollama by default).

This is the winning model from MTP Report Table 1 (Gemma3 7B: 98.5% accident
type accuracy, 5.2 s latency) and is therefore the natural local-deployment
choice for the Call Agent.
"""

from __future__ import annotations

import os
from typing import Dict, List

import urllib.request
import json

from .base import BaseLLM, LLMResult
from ..prompts import DISPATCHER_SYSTEM_PROMPT, EXTRACTION_PROMPT


class Gemma7BLLM(BaseLLM):
    name = "gemma3_7b"
    _model = "gemma3:7b"
    _host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def _request(self, prompt: str, system: str = "") -> str:
        payload = json.dumps({
            "model": self._model, "prompt": prompt, "system": system,
            "options": {"temperature": self.temperature},
            "stream": False,
        }).encode()
        req = urllib.request.Request(f"{self._host}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data.get("response", "")

    def _chat_real(self, system: str, history: List[Dict[str, str]],
                   user: str) -> LLMResult:
        conv = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        prompt = f"{conv}\nuser: {user}\nassistant:"
        text = self._request(prompt, system or DISPATCHER_SYSTEM_PROMPT)
        return LLMResult(text=text.strip())

    def _extract_real(self, conversation: str) -> LLMResult:
        prompt = EXTRACTION_PROMPT.format(conversation=conversation)
        text = self._request(prompt, "")
        return LLMResult(text=text, json_data=self._parse_json_from_text(text))
