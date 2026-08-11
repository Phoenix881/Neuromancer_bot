"""Hermes web extraction provider for Neuromancer."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any

from agent.web_search_provider import WebSearchProvider


LOGGER = logging.getLogger(__name__)
PROFILE_ROOT = Path(__file__).resolve().parents[2]
if str(PROFILE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROFILE_ROOT))

from neuromancer_bot.web_extract import DEFAULT_CHAR_LIMIT, DEFAULT_TIMEOUT_SECONDS, extract_documents, extract_for_tool


class NeuromancerReadableExtractProvider(WebSearchProvider):
    """Extraction-only backend that fetches public HTML/text pages directly."""

    @property
    def name(self) -> str:
        return "neuromancer-readable"

    @property
    def display_name(self) -> str:
        return "Neuromancer Readable Extract"

    def is_available(self) -> bool:
        return True

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    def extract(self, urls: list[str], **kwargs: Any) -> list[dict[str, Any]]:
        del kwargs
        return extract_documents(urls)

    def get_setup_schema(self) -> dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "local",
            "tag": "Free local HTML/text extraction for URLs found by DuckDuckGo search.",
            "env_vars": [],
        }


WEB_EXTRACT_READABLE_SCHEMA = {
    "name": "web_extract_readable",
    "description": (
        "Extract readable text from public web page URLs using Neuromancer's local "
        "HTML/text extractor. Use this after web_search when DuckDuckGo returns "
        "URLs/snippets and you need the actual page content. Handles ordinary "
        "HTML, text, markdown, XML, and JSON pages; it does not read PDFs or "
        "JavaScript-only pages."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Public HTTP(S) URLs to extract. Max 5 URLs.",
                "maxItems": 5,
            },
            "char_limit": {
                "type": "integer",
                "description": "Per-page character budget returned inline. Default 15000.",
                "minimum": 2000,
                "maximum": 50000,
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Per-page fetch timeout. Default 20 seconds.",
                "minimum": 3,
                "maximum": 60,
            },
        },
        "required": ["urls"],
    },
}


def _safe_urls(urls: Any) -> list[str]:
    if not isinstance(urls, list):
        return []
    safe: list[str] = []
    for item in urls[:5]:
        if isinstance(item, dict):
            item = item.get("url") or item.get("href")
        if isinstance(item, str) and item.strip():
            safe.append(item.strip())
    return safe


def _handle_extract_readable(args: dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    urls = _safe_urls(args.get("urls"))
    if not urls:
        return json.dumps(
            {"success": False, "error": "Pass one or more URL strings in urls."},
            ensure_ascii=False,
        )

    char_limit = args.get("char_limit", DEFAULT_CHAR_LIMIT)
    timeout_seconds = args.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    payload = extract_for_tool(urls, char_limit=char_limit, timeout_seconds=timeout_seconds)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def register(ctx) -> None:
    ctx.register_web_search_provider(NeuromancerReadableExtractProvider())
    ctx.register_tool(
        name="web_extract_readable",
        toolset="web",
        schema=WEB_EXTRACT_READABLE_SCHEMA,
        handler=_handle_extract_readable,
        description=WEB_EXTRACT_READABLE_SCHEMA["description"],
        emoji="R",
    )
    LOGGER.info("Registered Neuromancer readable web extraction")
