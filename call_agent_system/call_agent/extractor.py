"""
InformationExtractor - validates and normalises the JSON produced by an LLM.

Ensures that the downstream Resource Allocation Agent (MTP-1 dispatcher)
always receives a schema-compliant record, regardless of which backend
produced it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import REQUIRED_FIELDS


# Normalisation look-up tables -----------------------------------------------
_TYPE_ALIASES = {
    "car crash": "accident", "collision": "accident", "rta": "accident",
    "road accident": "accident", "accidents": "accident",
    "blaze": "fire", "flame": "fire",
    "heart attack": "medical", "injury": "medical",
    "robbery": "crime", "theft": "crime", "assault": "crime",
}
_SEVERITY_ALIASES = {
    "minor": "low", "moderate": "medium", "severe": "high",
    "catastrophic": "critical",
}
_SERVICE_ALIASES = {
    "ambulance": "ambulance", "police": "police", "cops": "police",
    "fire truck": "fire_truck", "fire_brigade": "fire_truck",
    "fire engine": "fire_truck", "rescue team": "rescue", "ndrf": "rescue",
}


class InformationExtractor:
    """Wraps a BaseLLM to guarantee a schema-valid result."""

    def __init__(self, llm):
        self.llm = llm

    def extract(self, conversation: str,
                ground_truth: Optional[Dict[str, Any]] = None,
                noisy: bool = False,
                asr_wer: float = 0.0) -> Dict[str, Any]:
        res = self.llm.extract(conversation, ground_truth=ground_truth,
                               noisy=noisy, asr_wer=asr_wer)
        data = res.json_data or self.llm._parse_json_from_text(res.text) or {}
        normed = self._normalise(data)
        normed["_latency"] = res.latency
        normed["_backend"] = res.backend
        return normed

    def _normalise(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for f in REQUIRED_FIELDS:
            out[f] = raw.get(f)

        if isinstance(out.get("type"), str):
            out["type"] = _TYPE_ALIASES.get(out["type"].lower(),
                                            out["type"].lower())
        if isinstance(out.get("severity"), str):
            out["severity"] = _SEVERITY_ALIASES.get(
                out["severity"].lower(), out["severity"].lower())

        services = out.get("services") or []
        if isinstance(services, str):
            services = [services]
        out["services"] = [
            _SERVICE_ALIASES.get(s.lower().replace(" ", "_"),
                                 s.lower().replace(" ", "_"))
            for s in services]

        if out.get("hazards") is None:
            out["hazards"] = []
        if out.get("victims") is None:
            out["victims"] = "0"
        elif isinstance(out["victims"], (int, float)):
            out["victims"] = str(int(out["victims"]))

        return out
