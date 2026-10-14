"""Minimal client for any OpenAI-compatible /chat/completions endpoint."""
import json
import re

import requests


class LLMError(Exception):
    pass


class LLMClient:
    def __init__(self, base_url, api_key, model, json_mode=True, timeout=30.0):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model or ""
        self.json_mode = json_mode
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def chat(self, messages, temperature=0.1, max_tokens=800, json_mode=None) -> str:
        if not self.configured:
            raise LLMError("LLM is not configured")
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if (self.json_mode if json_mode is None else json_mode):
            payload["response_format"] = {"type": "json_object"}
        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise LLMError(f"LLM request failed: {type(e).__name__}") from e
        if r.status_code != 200:
            # Never echo headers/key; include a short slice of the provider's error body.
            raise LLMError(f"LLM HTTP {r.status_code}: {r.text[:200]}")
        try:
            return r.json()["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise LLMError("LLM returned an unexpected response shape") from e


def parse_json_object(text: str):
    """Parse the first JSON object in a model reply (tolerates ```json fences / chatter)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        v = json.loads(text)
        return v if isinstance(v, dict) else None
    except ValueError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(text[start : i + 1])
                        if isinstance(v, dict):
                            return v
                    except ValueError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None
