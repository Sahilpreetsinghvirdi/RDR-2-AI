"""Ollama HTTP client for vision decisions (stdlib urllib only)."""

from __future__ import annotations

import json
import urllib.request


def ask_ollama(
    host: str,
    model: str,
    prompt: str,
    image_b64: str,
    timeout_s: float,
) -> str:
    """Send one image + prompt to /api/generate; return the raw text reply."""
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_b64],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    request = urllib.request.Request(
        host.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data = json.loads(response.read())
    text = data.get("response", "")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("empty reply from the vision model")
    return text
