"""Local readable-text extraction for URLs returned by web search."""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import sys
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


DEFAULT_TIMEOUT_SECONDS = 20
DEFAULT_MAX_BYTES = 2_000_000
DEFAULT_CHAR_LIMIT = 15_000
MAX_URLS = 5
USER_AGENT = (
    "Mozilla/5.0 (compatible; NeuromancerWebExtract/0.1; "
    "+https://github.com/karimabilgaziyev/neuromancer)"
)

_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
_SKIP_TAGS = {"script", "style", "noscript", "svg", "canvas", "template"}
_CREDENTIAL_QUERY_NAMES = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "authorization",
    "bearer",
    "client_secret",
    "cookie",
    "key",
    "password",
    "refresh_token",
    "secret",
    "session",
    "token",
}
_BLOCKED_HOSTS = {
    "0.0.0.0",
    "localhost",
    "metadata",
    "metadata.google.internal",
}


class ExtractionError(Exception):
    """Raised when a URL cannot be safely fetched or parsed."""


class _SafeRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        _validate_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = build_opener(_SafeRedirectHandler)


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._in_title = False
        self._title_parts: list[str] = []
        self._body_parts: list[str] = []

    @property
    def title(self) -> str:
        return _clean_text(" ".join(self._title_parts)).strip()

    @property
    def text(self) -> str:
        return _clean_text("".join(self._body_parts)).strip()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if self._skip_depth == 0 and tag in _BLOCK_TAGS:
            self._body_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
            return
        if self._skip_depth == 0 and tag in _BLOCK_TAGS:
            self._body_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not data or self._skip_depth > 0:
            return
        if self._in_title:
            self._title_parts.append(data)
        else:
            self._body_parts.append(data)


def extract_documents(
    urls: list[str],
    *,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[dict[str, Any]]:
    """Fetch and extract readable text from URLs.

    Returns Hermes web-provider compatible document dictionaries. Per-URL
    failures are returned as result rows with an ``error`` field.
    """

    safe_timeout = _bounded_int(
        timeout_seconds,
        default=DEFAULT_TIMEOUT_SECONDS,
        minimum=3,
        maximum=60,
    )
    safe_max_bytes = _bounded_int(
        max_bytes,
        default=DEFAULT_MAX_BYTES,
        minimum=100_000,
        maximum=5_000_000,
    )
    documents: list[dict[str, Any]] = []
    for url in urls[:MAX_URLS]:
        try:
            documents.append(
                _extract_one(url, timeout_seconds=safe_timeout, max_bytes=safe_max_bytes)
            )
        except Exception as exc:  # noqa: BLE001 - per-URL failure belongs in output
            documents.append(
                {
                    "url": str(url),
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "error": str(exc),
                    "metadata": {"sourceURL": str(url)},
                }
            )
    return documents


def extract_for_tool(
    urls: list[str],
    *,
    char_limit: int = DEFAULT_CHAR_LIMIT,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Return a compact JSON-serializable result for the direct Hermes tool."""

    safe_limit = _bounded_int(
        char_limit,
        default=DEFAULT_CHAR_LIMIT,
        minimum=2_000,
        maximum=50_000,
    )
    results: list[dict[str, Any]] = []
    for doc in extract_documents(urls, timeout_seconds=timeout_seconds):
        content = doc.get("content", "") or doc.get("raw_content", "")
        trimmed, truncated = _trim_text(str(content), safe_limit)
        item = {
            "url": doc.get("url", ""),
            "title": doc.get("title", ""),
            "content": trimmed,
            "error": doc.get("error"),
        }
        if truncated:
            item["truncated"] = True
        results.append(item)
    return {"results": results}


def _extract_one(url: str, *, timeout_seconds: int, max_bytes: int) -> dict[str, Any]:
    normalized_url = _validate_url(url)
    _check_website_policy(normalized_url)

    request = Request(
        normalized_url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,text/plain,application/json,*/*;q=0.8",
        },
    )
    try:
        with _OPENER.open(request, timeout=timeout_seconds) as response:
            final_url = response.geturl()
            _validate_url(final_url)
            status = getattr(response, "status", 200)
            headers = response.headers
            content_type = headers.get("Content-Type", "")
            raw = response.read(max_bytes + 1)
    except HTTPError as exc:
        raise ExtractionError(f"HTTP {exc.code} while fetching {normalized_url}") from exc
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise ExtractionError(f"Could not fetch {normalized_url}: {reason}") from exc

    was_byte_truncated = len(raw) > max_bytes
    raw = raw[:max_bytes]
    text = _decode_bytes(raw, content_type)
    title = ""
    if _is_html(content_type, text):
        title, content = _extract_html_text(text)
    elif "json" in content_type.lower():
        content = _format_json_text(text)
    elif _is_textual(content_type):
        content = _clean_text(text)
    else:
        raise ExtractionError(
            f"Unsupported content type {content_type or '<unknown>'}; local extractor handles HTML and text"
        )

    if not content:
        raise ExtractionError("No readable text found on page")
    if was_byte_truncated:
        content += f"\n\n[local fetch truncated at {max_bytes:,} bytes]"

    return {
        "url": final_url or normalized_url,
        "title": title or _fallback_title(final_url or normalized_url),
        "content": content,
        "raw_content": content,
        "metadata": {
            "sourceURL": normalized_url,
            "finalURL": final_url or normalized_url,
            "status": status,
            "contentType": content_type,
            "extractor": "neuromancer-readable",
        },
    }


def _validate_url(url: str) -> str:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"}:
        raise ExtractionError("Only http and https URLs can be extracted")
    if not parsed.hostname:
        raise ExtractionError("URL has no hostname")
    if parsed.username or parsed.password:
        raise ExtractionError("URLs with embedded usernames or passwords are blocked")
    normalized = parsed.geturl()
    hermes_sensitive_name = _hermes_sensitive_query_param_name(normalized)
    sensitive_name = hermes_sensitive_name or _sensitive_query_param_name(parsed.query)
    if sensitive_name:
        raise ExtractionError(f"Blocked URL with credential-like query parameter: {sensitive_name}")
    hermes_safe = _hermes_is_safe_url(normalized)
    if hermes_safe is False:
        raise ExtractionError("Blocked private or internal hostname")
    if hermes_safe is None and not _host_is_public_enough(parsed.hostname):
        raise ExtractionError("Blocked private or internal hostname")
    return normalized


def _host_is_public_enough(hostname: str) -> bool:
    host = hostname.strip().lower().rstrip(".")
    if not host:
        return False
    if host in _BLOCKED_HOSTS or host.endswith(".localhost") or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return ip.is_global


def _sensitive_query_param_name(query: str) -> str | None:
    for name, _value in parse_qsl(query, keep_blank_values=True):
        cleaned = name.strip().lower()
        if cleaned in _CREDENTIAL_QUERY_NAMES or cleaned.endswith("_token") or cleaned.endswith("_secret"):
            return name
    return None


def _hermes_sensitive_query_param_name(url: str) -> str | None:
    try:
        from tools.url_safety import sensitive_query_param_name

        return sensitive_query_param_name(url)
    except Exception:  # noqa: BLE001 - standalone mode
        return None


def _hermes_is_safe_url(url: str) -> bool | None:
    try:
        from tools.url_safety import is_safe_url

        return bool(is_safe_url(url))
    except Exception:  # noqa: BLE001 - standalone mode
        return None


def _check_website_policy(url: str) -> None:
    try:
        from tools.website_policy import check_website_access

        blocked = check_website_access(url)
    except Exception:  # noqa: BLE001 - standalone mode does not have Hermes tools
        return
    if blocked:
        raise ExtractionError(blocked.get("message") or "Blocked by website policy")


def _decode_bytes(raw: bytes, content_type: str) -> str:
    charset = "utf-8"
    match = re.search(r"charset=([^;\s]+)", content_type or "", flags=re.IGNORECASE)
    if match:
        charset = match.group(1).strip("\"'")
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


def _is_html(content_type: str, text: str) -> bool:
    lowered_type = (content_type or "").lower()
    return "html" in lowered_type or bool(re.search(r"<html|<!doctype html|<body", text[:1000], re.I))


def _is_textual(content_type: str) -> bool:
    lowered = (content_type or "").lower()
    if lowered.startswith("text/"):
        return True
    return any(token in lowered for token in ("markdown", "xml", "json", "javascript"))


def _extract_html_text(html: str) -> tuple[str, str]:
    parser = _ReadableHTMLParser()
    parser.feed(html)
    parser.close()
    return parser.title, parser.text


def _format_json_text(text: str) -> str:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return _clean_text(text)
    return json.dumps(parsed, ensure_ascii=False, indent=2)


def _clean_text(text: str) -> str:
    text = unescape(text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False
    return "\n".join(cleaned_lines).strip()


def _trim_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head_limit = int(limit * 0.8)
    tail_limit = limit - head_limit
    head = text[:head_limit].rstrip()
    tail = text[-tail_limit:].lstrip()
    return f"{head}\n\n[... middle omitted by local extractor ...]\n\n{tail}", True


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def _fallback_title(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("-", " ").replace("_", " ")
    return path.split("/")[-1] if path else parsed.hostname or url


def _json_default(value: Any) -> str:
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract readable text from web URLs.")
    parser.add_argument("urls", nargs="+", help="HTTP(S) URLs to extract")
    parser.add_argument("--char-limit", type=int, default=DEFAULT_CHAR_LIMIT)
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    payload = extract_for_tool(args.urls, char_limit=args.char_limit, timeout_seconds=args.timeout)
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=_json_default)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
