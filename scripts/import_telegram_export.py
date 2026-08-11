#!/usr/bin/env python3
"""Import a Telegram Desktop JSON export into Neuromancer's SQLite memory."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DB_PATH = "neuromancer_memory.db"
DEFAULT_EXPORT_PATH = "result.json"
DEFAULT_NEUROMANCER_USER_ID = "8967384440"


def flatten_text(value: Any) -> str:
    """Telegram exports rich text as either a string or a list of chunks."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(flatten_text(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(value, dict):
        return flatten_text(value.get("text", ""))
    return str(value)


def normalize_user_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if text.startswith("user"):
        text = text[4:]
    return text or None


def sent_at_from_message(message: dict[str, Any]) -> str:
    unix_time = message.get("date_unixtime")
    if unix_time:
        return datetime.fromtimestamp(int(unix_time), tz=timezone.utc).isoformat()

    date_text = message.get("date")
    if not date_text:
        raise ValueError(f"Message {message.get('id')} has no date or date_unixtime")
    return datetime.fromisoformat(date_text).isoformat()


def media_descriptor(message: dict[str, Any], message_type: str) -> str:
    if message_type.startswith("service"):
        actor = message.get("actor") or "unknown"
        action = message.get("action") or "service"
        members = message.get("members") or []
        suffix = f": {', '.join(map(str, members))}" if members else ""
        return f"[service] {actor} {action}{suffix}"

    if message_type == "text":
        return ""

    if message_type == "sticker":
        emoji = message.get("sticker_emoji")
        return f"[sticker {emoji}]" if emoji else "[sticker]"

    file_name = message.get("file_name")
    if file_name:
        return f"[{message_type}: {file_name}]"
    return f"[{message_type}]"


def classify_message(message: dict[str, Any]) -> str:
    if message.get("type") == "service":
        action = message.get("action") or "service"
        return f"service:{action}"
    if message.get("media_type"):
        return str(message["media_type"])
    if message.get("photo") is not None:
        return "photo"
    if message.get("file") is not None:
        return "file"
    return "text"


def text_for_context(message: dict[str, Any], message_type: str) -> str:
    text = flatten_text(message.get("text"))
    descriptor = media_descriptor(message, message_type)
    if not descriptor:
        return text
    if text.strip():
        return f"{descriptor} {text}"
    return descriptor


def derive_chat_id(export_data: dict[str, Any], db_chat_id: str | None) -> str:
    if db_chat_id:
        return db_chat_id

    export_id = export_data.get("id")
    if export_id is None:
        raise ValueError("Export has no top-level chat id and DB has no group_profile row")

    export_type = str(export_data.get("type", ""))
    if export_type in {"private_supergroup", "public_supergroup", "private_channel", "public_channel"}:
        return f"-100{export_id}"
    return str(export_id)


def load_db_chat_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT chat_id FROM group_profile WHERE id = 1").fetchone()
    return str(row[0]) if row else None


def ensure_member(
    conn: sqlite3.Connection,
    *,
    chat_id: str,
    user_id: str,
    display_name: str,
    is_bot: bool,
    is_neuromancer: bool,
) -> None:
    if is_neuromancer:
        profile = (
            "Neuromancer, current bot identity. Imported Telegram export messages "
            "from user 8967384440 are treated as this Neuromancer's own messages."
        )
    else:
        profile = f"Imported from Telegram export as {display_name}."

    conn.execute(
        """
        INSERT INTO members (
          user_id, chat_id, username, primary_name, is_bot, profile, raw_profile
        )
        VALUES (?, ?, NULL, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          chat_id = excluded.chat_id,
          is_bot = CASE WHEN excluded.is_bot = 1 THEN 1 ELSE members.is_bot END,
          updated_at = CURRENT_TIMESTAMP
        """,
        (user_id, chat_id, display_name or user_id, int(is_bot), profile, profile),
    )

    if display_name:
        conn.execute(
            """
            INSERT OR IGNORE INTO member_aliases (
              user_id, alias, normalized_alias, source
            )
            VALUES (?, ?, lower(?), 'import')
            """,
            (user_id, display_name, display_name),
        )


def import_messages(
    conn: sqlite3.Connection,
    export_data: dict[str, Any],
    *,
    chat_id: str,
    neuromancer_user_id: str,
) -> tuple[int, int, int]:
    messages = export_data.get("messages")
    if not isinstance(messages, list):
        raise ValueError("Export JSON does not contain a messages array")

    neuromancer_reply_targets = {
        str(message["reply_to_message_id"])
        for message in messages
        if normalize_user_id(message.get("from_id")) == neuromancer_user_id
        and message.get("reply_to_message_id") is not None
    }

    known_bot_rows = conn.execute("SELECT user_id FROM members WHERE is_bot = 1").fetchall()
    known_bot_user_ids = {str(row[0]) for row in known_bot_rows}
    known_bot_user_ids.add(neuromancer_user_id)

    imported = 0
    skipped = 0
    imported_user_ids: set[str] = set()

    for message in messages:
        telegram_message_id = str(message.get("id") or "")
        if not telegram_message_id:
            skipped += 1
            continue

        sender_user_id = normalize_user_id(message.get("from_id")) or normalize_user_id(message.get("actor_id"))
        sender_name = message.get("from") or message.get("actor")
        sender_name_snapshot = str(sender_name) if sender_name is not None else None
        is_neuromancer = sender_user_id == neuromancer_user_id
        is_bot = is_neuromancer or (sender_user_id in known_bot_user_ids)

        if sender_user_id:
            ensure_member(
                conn,
                chat_id=chat_id,
                user_id=sender_user_id,
                display_name=sender_name_snapshot or sender_user_id,
                is_bot=is_bot,
                is_neuromancer=is_neuromancer,
            )
            imported_user_ids.add(sender_user_id)

        message_type = classify_message(message)
        context_text = text_for_context(message, message_type)
        raw_json = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        trigger_text = flatten_text(message.get("text")).lower()
        was_trigger = (
            telegram_message_id in neuromancer_reply_targets
            or "/ai" in trigger_text
            or "@neuromancer" in trigger_text
        )

        conn.execute(
            """
            INSERT INTO messages_raw (
              chat_id,
              telegram_message_id,
              sender_user_id,
              sender_name_snapshot,
              sent_at,
              text,
              message_type,
              reply_to_message_id,
              is_bot_message,
              is_neuromancer_message,
              was_trigger,
              raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id, telegram_message_id) DO UPDATE SET
              sender_user_id = excluded.sender_user_id,
              sender_name_snapshot = excluded.sender_name_snapshot,
              sent_at = excluded.sent_at,
              text = excluded.text,
              message_type = excluded.message_type,
              reply_to_message_id = excluded.reply_to_message_id,
              is_bot_message = excluded.is_bot_message,
              is_neuromancer_message = excluded.is_neuromancer_message,
              was_trigger = excluded.was_trigger,
              raw_json = excluded.raw_json
            """,
            (
                chat_id,
                telegram_message_id,
                sender_user_id,
                sender_name_snapshot,
                sent_at_from_message(message),
                context_text,
                message_type,
                str(message["reply_to_message_id"]) if message.get("reply_to_message_id") is not None else None,
                int(is_bot),
                int(is_neuromancer),
                int(was_trigger),
                raw_json,
            ),
        )
        imported += 1

    for user_id in sorted(imported_user_ids):
        row = conn.execute(
            """
            SELECT MIN(sent_at), MAX(sent_at), COUNT(*)
            FROM messages_raw
            WHERE chat_id = ? AND sender_user_id = ?
            """,
            (chat_id, user_id),
        ).fetchone()
        if not row or not row[0]:
            continue
        first_seen_at, last_seen_at, message_count = row
        conn.execute(
            """
            UPDATE members
            SET
              first_seen_at = CASE
                WHEN first_seen_at IS NULL OR ? < first_seen_at THEN ?
                ELSE first_seen_at
              END,
              last_seen_at = CASE
                WHEN last_seen_at IS NULL OR ? > last_seen_at THEN ?
                ELSE last_seen_at
              END,
              message_count = ?,
              updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """,
            (first_seen_at, first_seen_at, last_seen_at, last_seen_at, message_count, user_id),
        )

    return imported, skipped, len(imported_user_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate neuromancer_memory.db from a Telegram Desktop JSON export."
    )
    parser.add_argument("export_json", nargs="?", default=DEFAULT_EXPORT_PATH)
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="Path to neuromancer_memory.db")
    parser.add_argument(
        "--neuromancer-user-id",
        default=DEFAULT_NEUROMANCER_USER_ID,
        help="Telegram user id whose messages should be treated as Neuromancer/self",
    )
    parser.add_argument(
        "--chat-id",
        help="Override chat id. Defaults to group_profile.chat_id, then derives from export metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_path = Path(args.export_json)
    db_path = Path(args.db)

    with export_path.open("r", encoding="utf-8") as file:
        export_data = json.load(file)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        chat_id = args.chat_id or derive_chat_id(export_data, load_db_chat_id(conn))
        imported, skipped, users = import_messages(
            conn,
            export_data,
            chat_id=chat_id,
            neuromancer_user_id=str(args.neuromancer_user_id),
        )
        conn.commit()

    print(
        f"Imported {imported} messages into {db_path} for chat {chat_id}; "
        f"skipped {skipped}; touched {users} users."
    )


if __name__ == "__main__":
    main()
