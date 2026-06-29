"""Google Gemini Flash 2.5 Pro backend (the default choice in the MTP report)."""

from __future__ import annotations

import os
from typing import Dict, List

from .base import BaseLLM, LLMResult
from ..prompts import DISPATCHER_SYSTEM_PROMPT, EXTRACTION_PROMPT


class GeminiFlashLLM(BaseLLM):
    name = "gemini_flash_2_5"
    _model = "gemini-2.5-flash"

    def _client(self):
        try:
            import google.generativeai as genai              # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "google-generativeai package required for mode='real'. "
                "pip install google-generativeai") from exc
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        return genai

    def _chat_real(self, system: str, history: List[Dict[str, str]],
                   user: str) -> LLMResult:
        genai = self._client()
        model = genai.GenerativeModel(self._model,
                                      system_instruction=system or DISPATCHER_SYSTEM_PROMPT)
        chat = model.start_chat(history=[
            {"role": m["role"], "parts": [m["content"]]} for m in history])
        resp = chat.send_message(user, generation_config={
            "temperature": self.temperature, "max_output_tokens": 120})
        return LLMResult(text=resp.text)

    def _extract_real(self, conversation: str) -> LLMResult:
        genai = self._client()
        model = genai.GenerativeModel(self._model)
        resp = model.generate_content(
            EXTRACTION_PROMPT.format(conversation=conversation),
            generation_config={"temperature": 0.0, "max_output_tokens": 400,
                               "response_mime_type": "application/json"})
        return LLMResult(text=resp.text,
                         json_data=self._parse_json_from_text(resp.text))
