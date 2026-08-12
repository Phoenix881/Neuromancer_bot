"""Read-only search helpers for Neuromancer's SQLite message memory."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any


DEFAULT_DB_FILENAME = "neuromancer_memory.db"
DEFAULT_LIMIT = 10
MAX_LIMIT = 30
DEFAULT_CONTEXT_WINDOW = 5
MAX_CONTEXT_WINDOW = 20
DEFAULT_TEXT_CHARS = 900
MAX_TEXT_CHARS = 2_500


def search_messages(
    query: str = "",
    *,
    mode: str = "auto",
    limit: int = DEFAULT_LIMIT,
    sender_user_id: str | None = None,
    after: str | None = None,
    before: str | None = None,
    chat_id: str | None = None,
    db_path: Path | None = None,
    max_text_chars: int = DEFAULT_TEXT_CHARS,
) -> dict[str, Any]:
    """Search messages by FTS terms, exact substring, or message id."""

    safe_mode = normalize_mode(mode)
    safe_limit = bounded_int(limit, default=DEFAULT_LIMIT, minimum=1, maximum=MAX_LIMIT)
    safe_text_chars = bounded_int(
        max_text_chars,
        default=DEFAULT_TEXT_CHARS,
        minimum=120,
        maximum=MAX_TEXT_CHARS,
    )

    with readonly_connection(db_path) as conn:
        target_chat_id = chat_id or load_target_chat_id(conn)
        if safe_mode == "id" or (safe_mode == "auto" and looks_like_message_id(query)):
            lookup = lookup_message(
                str(query),
                chat_id=target_chat_id,
                db_path=db_path,
                max_text_chars=safe_text_chars,
            )
            return {
                "ok": True,
                "mode": "id",
                "query": query,
                "results": ([lookup["message"]] if lookup.get("message") else []),
            }

        if not str(query).strip() and not sender_user_id:
            return {"ok": False, "error": "query or sender_user_id is required", "results": []}

        rows: list[sqlite3.Row]
        used_mode = safe_mode
        if safe_mode in {"auto", "fts"} and str(query).strip():
            try:
                rows = search_fts_rows(
                    conn,
                    str(query),
                    chat_id=target_chat_id,
                    limit=safe_limit,
                    sender_user_id=sender_user_id,
                    after=after,
                    before=before,
                )
                used_mode = "fts"
            except sqlite3.Error as exc:
                if safe_mode == "fts":
                    return {"ok": False, "error": f"FTS search failed: {exc}", "results": []}
                rows = search_exact_rows(
                    conn,
                    str(query),
                    chat_id=target_chat_id,
                    limit=safe_limit,
                    sender_user_id=sender_user_id,
                    after=after,
                    before=before,
                )
                used_mode = "exact"
        else:
            rows = search_exact_rows(
                conn,
                str(query),
                chat_id=target_chat_id,
                limit=safe_limit,
                sender_user_id=sender_user_id,
                after=after,
                before=before,
            )
            used_mode = "exact"

        attachments = load_attachments_for_rows(conn, [int(row["id"]) for row in rows])
        return {
            "ok": True,
            "mode": used_mode,
            "query": query,
            "count": len(rows),
            "results": [
                row_to_message(row, attachments.get(int(row["id"]), []), max_text_chars=safe_text_chars)
                for row in rows
            ],
        }


def lookup_message(
    message_id: str,
    *,
    id_type: str = "auto",
    context_window: int = 0,
    chat_id: str | None = None,
    db_path: Path | None = None,
    max_text_chars: int = DEFAULT_TEXT_CHARS,
) -> dict[str, Any]:
    """Look up one message by Telegram id or raw DB id."""

    safe_window = bounded_int(
        context_window,
        default=0,
        minimum=0,
        maximum=MAX_CONTEXT_WINDOW,
    )
    safe_text_chars = bounded_int(
        max_text_chars,
        default=DEFAULT_TEXT_CHARS,
        minimum=120,
        maximum=MAX_TEXT_CHARS,
    )

    with readonly_connection(db_path) as conn:
        target_chat_id = chat_id or load_target_chat_id(conn)
        center = find_message_row(conn, str(message_id), target_chat_id, id_type=id_type)
        if center is None:
            return {
                "ok": False,
                "error": f"message not found: {message_id}",
                "message": None,
                "context": [],
            }

        context_rows = surrounding_rows(conn, center, target_chat_id, safe_window)
        attachments = load_attachments_for_rows(conn, [int(row["id"]) for row in context_rows])
        messages = [
            row_to_message(row, attachments.get(int(row["id"]), []), max_text_chars=safe_text_chars)
            for row in context_rows
        ]
        center_message = next(
            (message for message in messages if message["raw_id"] == int(center["id"])),
            row_to_message(center, attachments.get(int(center["id"]), []), max_text_chars=safe_text_chars),
        )
        return {
            "ok": True,
            "message": center_message,
            "context_window": safe_window,
            "context": messages,
        }


def recent_user_messages(
    sender_user_id: str,
    *,
    limit: int = DEFAULT_LIMIT,
    chat_id: str | None = None,
    db_path: Path | None = None,
    max_text_chars: int = DEFAULT_TEXT_CHARS,
) -> dict[str, Any]:
    """Return recent messages from one user id."""

    return search_messages(
        "",
        mode="exact",
        limit=limit,
        sender_user_id=str(sender_user_id),
        chat_id=chat_id,
        db_path=db_path,
        max_text_chars=max_text_chars,
    )


def readonly_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = resolve_db_path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"memory database not found: {path}")
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def resolve_db_path(db_path: Path | None = None) -> Path:
    if db_path is not None:
        path = Path(db_path)
    else:
        hermes_home = Path(os.environ.get("HERMES_HOME") or Path(__file__).resolve().parents[1])
        path = Path(os.environ.get("NEUROMANCER_DB_PATH") or hermes_home / DEFAULT_DB_FILENAME)
    if not path.is_absolute():
        path = Path(os.environ.get("HERMES_HOME") or Path.cwd()) / path
    return path.resolve()


def load_target_chat_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT chat_id FROM group_profile WHERE id = 1").fetchone()
    if row is None or row["chat_id"] is None:
        raise RuntimeError("group_profile row with id=1 is missing")
    return str(row["chat_id"])


def search_fts_rows(
    conn: sqlite3.Connection,
    query: str,
    *,
    chat_id: str,
    limit: int,
    sender_user_id: str | None,
    after: str | None,
    before: str | None,
) -> list[sqlite3.Row]:
    match_query = build_fts_query(query)
    clauses = ["m.chat_id = ?", "messages_fts MATCH ?"]
    params: list[Any] = [chat_id, match_query]
    append_optional_filters(clauses, params, sender_user_id=sender_user_id, after=after, before=before)
    params.append(limit)
    return conn.execute(
        f"""
        SELECT m.*
        FROM messages_fts
        JOIN messages_raw m ON m.id = messages_fts.rowid
        WHERE {" AND ".join(clauses)}
        ORDER BY m.sent_at DESC, m.id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()


def search_exact_rows(
    conn: sqlite3.Connection,
    query: str,
    *,
    chat_id: str,
    limit: int,
    sender_user_id: str | None,
    after: str | None,
    before: str | None,
) -> list[sqlite3.Row]:
    clauses = ["chat_id = ?"]
    params: list[Any] = [chat_id]
    if query.strip():
        pattern = f"%{escape_like(query.strip())}%"
        clauses.append(
            "(COALESCE(text, '') LIKE ? ESCAPE '\\' OR COALESCE(sender_name_snapshot, '') LIKE ? ESCAPE '\\')"
        )
        params.extend([pattern, pattern])
    append_optional_filters(clauses, params, sender_user_id=sender_user_id, after=after, before=before)
    params.append(limit)
    return conn.execute(
        f"""
        SELECT *
        FROM messages_raw
        WHERE {" AND ".join(clauses)}
        ORDER BY sent_at DESC, id DESC
        LIMIT ?
        """,
        params,
    ).fetchall()


def find_message_row(
    conn: sqlite3.Connection,
    message_id: str,
    chat_id: str,
    *,
    id_type: str,
) -> sqlite3.Row | None:
    cleaned = message_id.strip()
    lowered = cleaned.casefold()
    if lowered.startswith(("raw:", "db:")):
        id_type = "raw"
        cleaned = cleaned.split(":", 1)[1].strip()
    elif lowered.startswith(("tg:", "telegram:")):
        id_type = "telegram"
        cleaned = cleaned.split(":", 1)[1].strip()

    if id_type not in {"auto", "raw", "telegram"}:
        id_type = "auto"

    if id_type in {"auto", "telegram"}:
        row = conn.execute(
            """
            SELECT *
            FROM messages_raw
            WHERE chat_id = ? AND telegram_message_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (chat_id, cleaned),
        ).fetchone()
        if row is not None or id_type == "telegram":
            return row

    if id_type in {"auto", "raw"} and cleaned.isdigit():
        return conn.execute(
            """
            SELECT *
            FROM messages_raw
            WHERE chat_id = ? AND id = ?
            LIMIT 1
            """,
            (chat_id, int(cleaned)),
        ).fetchone()
    return None


def surrounding_rows(
    conn: sqlite3.Connection,
    center: sqlite3.Row,
    chat_id: str,
    window: int,
) -> list[sqlite3.Row]:
    if window <= 0:
        return [center]

    before = conn.execute(
        """
        SELECT *
        FROM messages_raw
        WHERE chat_id = ?
          AND (sent_at < ? OR (sent_at = ? AND id < ?))
        ORDER BY sent_at DESC, id DESC
        LIMIT ?
        """,
        (chat_id, center["sent_at"], center["sent_at"], center["id"], window),
    ).fetchall()
    after = conn.execute(
        """
        SELECT *
        FROM messages_raw
        WHERE chat_id = ?
          AND (sent_at > ? OR (sent_at = ? AND id > ?))
        ORDER BY sent_at ASC, id ASC
        LIMIT ?
        """,
        (chat_id, center["sent_at"], center["sent_at"], center["id"], window),
    ).fetchall()
    return list(reversed(before)) + [center] + list(after)


def load_attachments_for_rows(
    conn: sqlite3.Connection,
    raw_ids: list[int],
) -> dict[int, list[dict[str, Any]]]:
    if not raw_ids:
        return {}
    placeholders = ",".join("?" for _ in raw_ids)
    try:
        rows = conn.execute(
            f"""
            SELECT
              message_raw_id,
              media_type,
              mime_type,
              width,
              height,
              file_size,
              vision_status,
              vision_summary,
              vision_ocr
            FROM message_attachments
            WHERE message_raw_id IN ({placeholders})
            ORDER BY id ASC
            """,
            raw_ids,
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["message_raw_id"]), []).append(
            {
                "media_type": row["media_type"],
                "mime_type": row["mime_type"],
                "width": row["width"],
                "height": row["height"],
                "file_size": row["file_size"],
                "vision_status": row["vision_status"],
                "vision_summary": clamp_text(row["vision_summary"] or "", 500),
                "vision_ocr": clamp_text(row["vision_ocr"] or "", 500),
            }
        )
    return grouped


def row_to_message(
    row: sqlite3.Row,
    attachments: list[dict[str, Any]],
    *,
    max_text_chars: int,
) -> dict[str, Any]:
    return {
        "raw_id": int(row["id"]),
        "telegram_message_id": str(row["telegram_message_id"]),
        "sent_at": str(row["sent_at"]),
        "sender_user_id": str(row["sender_user_id"]) if row["sender_user_id"] is not None else None,
        "sender_name": row["sender_name_snapshot"],
        "text": clamp_text(row["text"] or "", max_text_chars),
        "message_type": row["message_type"],
        "reply_to_message_id": row["reply_to_message_id"],
        "is_bot_message": bool(row["is_bot_message"]),
        "is_neuromancer_message": bool(row["is_neuromancer_message"]),
        "was_trigger": bool(row["was_trigger"]),
        "attachments": attachments,
    }


def append_optional_filters(
    clauses: list[str],
    params: list[Any],
    *,
    sender_user_id: str | None,
    after: str | None,
    before: str | None,
) -> None:
    if sender_user_id:
        clauses.append("sender_user_id = ?")
        params.append(str(sender_user_id))
    if after:
        clauses.append("sent_at >= ?")
        params.append(str(after))
    if before:
        clauses.append("sent_at <= ?")
        params.append(str(before))


def build_fts_query(query: str) -> str:
    stripped = query.strip()
    if not stripped:
        raise sqlite3.OperationalError("empty FTS query")
    if len(stripped) >= 2 and stripped[0] == stripped[-1] == '"':
        phrase = stripped[1:-1].strip()
        if not phrase:
            raise sqlite3.OperationalError("empty FTS phrase")
        return quote_fts(phrase)
    tokens = re.findall(r"[\w@.#+'-]+", stripped, flags=re.UNICODE)
    if not tokens:
        return quote_fts(stripped)
    return " AND ".join(quote_fts(token) for token in tokens)


def quote_fts(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def looks_like_message_id(value: str) -> bool:
    cleaned = str(value).strip().casefold()
    if cleaned.startswith(("raw:", "db:", "tg:", "telegram:")):
        cleaned = cleaned.split(":", 1)[1].strip()
    return cleaned.isdigit()


def normalize_mode(mode: str) -> str:
    value = str(mode or "auto").strip().casefold()
    return value if value in {"auto", "fts", "exact", "id"} else "auto"


def bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def clamp_text(value: str, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 28)].rstrip() + " ... [truncated]"


def json_dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only search in Neuromancer message memory.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    search_parser = subcommands.add_parser("search")
    search_parser.add_argument("query", nargs="?", default="")
    search_parser.add_argument("--mode", choices=("auto", "fts", "exact", "id"), default="auto")
    search_parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    search_parser.add_argument("--sender-user-id")

    context_parser = subcommands.add_parser("context")
    context_parser.add_argument("message_id")
    context_parser.add_argument("--id-type", choices=("auto", "raw", "telegram"), default="auto")
    context_parser.add_argument("--window", type=int, default=DEFAULT_CONTEXT_WINDOW)

    args = parser.parse_args(argv)
    if args.command == "search":
        payload = search_messages(
            args.query,
            mode=args.mode,
            limit=args.limit,
            sender_user_id=args.sender_user_id,
        )
    else:
        payload = lookup_message(
            args.message_id,
            id_type=args.id_type,
            context_window=args.window,
        )
    sys.stdout.write(json_dump(payload) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
