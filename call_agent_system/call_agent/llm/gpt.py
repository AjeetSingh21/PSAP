"""OpenAI GPT-4o backend."""

from __future__ import annotations

import os
from typing import Dict, List

from .base import BaseLLM, LLMResult
from ..prompts import DISPATCHER_SYSTEM_PROMPT, EXTRACTION_PROMPT


class GPT4oLLM(BaseLLM):
    name = "gpt4o"
    _model = "gpt-4o"

    def _client(self):
        try:
            from openai import OpenAI                       # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "openai package required for mode='real'. "
                "pip install openai") from exc
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _chat_real(self, system: str, history: List[Dict[str, str]],
                   user: str) -> LLMResult:
        client = self._client()
        messages = [{"role": "system", "content": system or DISPATCHER_SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user})
        resp = client.chat.completions.create(
            model=self._model, messages=messages,
            temperature=self.temperature, max_tokens=120)
        text = resp.choices[0].message.content
        return LLMResult(text=text, tokens=resp.usage.completion_tokens)

    def _extract_real(self, conversation: str) -> LLMResult:
        client = self._client()
        prompt = EXTRACTION_PROMPT.format(conversation=conversation)
        resp = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content":
                       "You extract JSON from PSAP call transcripts."},
                      {"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=400,
            response_format={"type": "json_object"})
        txt = resp.choices[0].message.content
        return LLMResult(text=txt, json_data=self._parse_json_from_text(txt),
                         tokens=resp.usage.completion_tokens)
