"""OpenRouter vision preprocessing for Telegram images."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VISION_PROMPT = """Analyze this Telegram image for Neuromancer's chat memory.

Return only a valid JSON object with these keys:
- summary: concise description of what is visually present and what matters in chat
- ocr: readable text in the image, or empty string
- vibe: tone/context, especially if it is a meme, screenshot, document, or reaction image
- details: short list of important visual details

Do not invent identities. If text is unclear, say so."""


class VisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class VisionResult:
    summary: str
    ocr: str
    raw_json: str


class OpenRouterVisionClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = OPENROUTER_URL,
        timeout_seconds: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def analyze_image(self, path: Path) -> VisionResult:
        api_key = openrouter_api_key()
        if not api_key:
            raise VisionError("OPENROUTER_API_KEY is not set")

        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        image_data = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 700,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/neuromancer-local",
                "X-Title": "Neuromancer Telegram Vision",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise VisionError(f"OpenRouter vision HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise VisionError(f"OpenRouter vision request failed: {exc}") from exc

        return parse_openrouter_response(raw_response)


def openrouter_api_key() -> str | None:
    for name in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "HERMES_OPENROUTER_API_KEY",
        "OPENROUTER_API_TOKEN",
    ):
        value = os.environ.get(name)
        if value:
            return value
    return None


def parse_openrouter_response(raw_response: str) -> VisionResult:
    try:
        data = json.loads(raw_response)
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise VisionError("OpenRouter vision response was not in the expected format") from exc

    content_text = content_to_text(content)
    parsed = parse_jsonish_content(content_text)
    if isinstance(parsed, dict):
        summary = str(parsed.get("summary") or "").strip()
        ocr = str(parsed.get("ocr") or "").strip()
        raw_json = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return VisionResult(summary=summary or content_text.strip(), ocr=ocr, raw_json=raw_json)

    text = content_text.strip()
    return VisionResult(summary=text, ocr="", raw_json=json.dumps({"summary": text}, ensure_ascii=False))


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text")
                if value:
                    parts.append(str(value))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def parse_jsonish_content(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None
