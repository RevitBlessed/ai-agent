import json
import re
from typing import Dict, Optional, List, Any


_FENCED_JSON_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_tool_payload(text: str) -> Optional[Dict]:
    if not text:
        return None
    text = text.strip()

    fenced = _FENCED_JSON_RE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return None

    brace_index = text.find("{")
    if brace_index == -1:
        return None

    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(text[brace_index:])
        return payload
    except json.JSONDecodeError:
        return None


def extract_tool_payloads(text: str) -> List[Dict]:
    if not text:
        return []
    text = text.strip()

    fenced = _FENCED_JSON_RE.findall(text)
    payloads = []
    for block in fenced:
        try:
            payloads.append(json.loads(block))
        except json.JSONDecodeError:
            continue
    if payloads:
        return payloads

    single = extract_tool_payload(text)
    return [single] if single else []


def extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    fenced = _FENCED_JSON_RE.search(text)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            return None
    brace_index = text.find("{")
    if brace_index == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(text[brace_index:])
        return payload
    except json.JSONDecodeError:
        return None
