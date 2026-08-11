"""Neuromancer Telegram memory recorder.

This plugin is deliberately write-narrow:
- it appends/upserts rows in messages_raw;
- it updates member aliases and first/last/message_count aggregates;
- it never updates group_profile.profile/raw_profile or members.profile/raw_profile.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from hermes_cli.config import get_hermes_home


LOGGER = logging.getLogger(__name__)
DB_FILENAME = "neuromancer_memory.db"
LEGACY_NEUROMANCER_USER_ID = "8967384440"


def _db_path() -> Path:
    return get_hermes_home() / DB_FILENAME


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=5)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if value:
        try:
            return datetime.fromisoformat(str(value)).isoformat()
        except ValueError:
            return str(value)
    return _utc_now()


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return _to_iso(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except Exception:
            pass
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_json_default)


def _normalized_alias(value: str) -> str:
    return value.strip().casefold()


def _load_target_chat_id(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT chat_id FROM group_profile WHERE id = 1").fetchone()
    return str(row[0]) if row and row[0] is not None else None


def _chat_id_from_message(message: Any) -> str | None:
    chat = getattr(message, "chat", None)
    chat_id = getattr(chat, "id", None)
    return str(chat_id) if chat_id is not None else None


def _chat_id_from_event(event: Any) -> str | None:
    source = getattr(event, "source", None)
    chat_id = getattr(source, "chat_id", None)
    return str(chat_id) if chat_id is not None else None


def _target_chat_matches(conn: sqlite3.Connection, chat_id: Any) -> bool:
    target = _load_target_chat_id(conn)
    return bool(target and chat_id is not None and str(chat_id) == target)


def _display_name_from_user(user: Any) -> str:
    for attr in ("full_name", "name"):
        value = getattr(user, attr, None)
        if value:
            return str(value)
    first = str(getattr(user, "first_name", "") or "").strip()
    last = str(getattr(user, "last_name", "") or "").strip()
    combined = " ".join(part for part in (first, last) if part)
    if combined:
        return combined
    username = getattr(user, "username", None)
    return str(username) if username else ""


def _user_from_message(message: Any) -> tuple[str | None, str | None, str | None, bool]:
    user = getattr(message, "from_user", None)
    if user is None:
        return None, None, None, False
    user_id = getattr(user, "id", None)
    username = getattr(user, "username", None)
    display_name = _display_name_from_user(user)
    is_bot = bool(getattr(user, "is_bot", False))
    return (
        str(user_id) if user_id is not None else None,
        display_name or None,
        str(username) if username else None,
        is_bot,
    )


def _bot_identity(adapter: Any) -> tuple[str, str, str | None]:
    bot = getattr(adapter, "_bot", None)
    bot_id = getattr(bot, "id", None) if bot is not None else None
    username = getattr(bot, "username", None) if bot is not None else None
    first_name = getattr(bot, "first_name", None) if bot is not None else None
    display = str(first_name or "Neuromancer")
    return (
        str(bot_id) if bot_id is not None else LEGACY_NEUROMANCER_USER_ID,
        display,
        str(username) if username else None,
    )


def _is_neuromancer_user(user_id: str | None, adapter: Any = None) -> bool:
    if not user_id:
        return False
    if user_id == LEGACY_NEUROMANCER_USER_ID:
        return True
    bot = getattr(adapter, "_bot", None) if adapter is not None else None
    bot_id = getattr(bot, "id", None) if bot is not None else None
    return bot_id is not None and str(bot_id) == user_id


def _aliases_for(display_name: str | None, username: str | None) -> list[str]:
    aliases: list[str] = []
    for value in (display_name, username, f"@{username}" if username else None):
        text = str(value).strip() if value else ""
        if text and text not in aliases:
            aliases.append(text)
    return aliases


def _ensure_member(
    conn: sqlite3.Connection,
    *,
    user_id: str,
    chat_id: str,
    display_name: str | None,
    username: str | None,
    is_bot: bool,
    is_neuromancer: bool = False,
) -> None:
    primary = display_name or username or user_id
    if is_neuromancer:
        profile = (
            "Neuromancer, current bot identity. Live Telegram messages are "
            "recorded by the neuromancer-memory Hermes plugin."
        )
    else:
        profile = f"Live Telegram member first seen as {primary}."

    # On conflict, intentionally avoid changing profile/raw_profile/primary_name.
    conn.execute(
        """
        INSERT INTO members (
          user_id, chat_id, username, primary_name, is_bot, profile, raw_profile
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          chat_id = excluded.chat_id,
          username = COALESCE(members.username, excluded.username),
          is_bot = CASE WHEN excluded.is_bot = 1 THEN 1 ELSE members.is_bot END,
          updated_at = CURRENT_TIMESTAMP
        """,
        (user_id, chat_id, username, primary, int(is_bot), profile, profile),
    )

    for alias in _aliases_for(display_name, username):
        conn.execute(
            """
            INSERT OR IGNORE INTO member_aliases (
              user_id, alias, normalized_alias, source
            )
            VALUES (?, ?, ?, 'live')
            """,
            (user_id, alias, _normalized_alias(alias)),
        )


def _refresh_member_stats(conn: sqlite3.Connection, chat_id: str, user_id: str | None) -> None:
    if not user_id:
        return
    row = conn.execute(
        """
        SELECT MIN(sent_at), MAX(sent_at), COUNT(*)
        FROM messages_raw
        WHERE chat_id = ? AND sender_user_id = ?
        """,
        (chat_id, user_id),
    ).fetchone()
    if not row or not row[0]:
        return
    first_seen_at, last_seen_at, message_count = row
    conn.execute(
        """
        UPDATE members
        SET
          first_seen_at = ?,
          last_seen_at = ?,
          message_count = ?,
          updated_at = CURRENT_TIMESTAMP
        WHERE user_id = ?
        """,
        (first_seen_at, last_seen_at, message_count, user_id),
    )


def _raw_message_payload(message: Any) -> dict[str, Any]:
    if hasattr(message, "to_dict") and callable(message.to_dict):
        try:
            data = message.to_dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    payload: dict[str, Any] = {"repr": repr(message)}
    for attr in ("message_id", "date", "text", "caption"):
        value = getattr(message, attr, None)
        if value is not None:
            payload[attr] = value
    return payload


def _message_type_value(message: Any, msg_type: Any) -> str:
    raw = getattr(msg_type, "value", None)
    if raw:
        return str(raw)
    for attr, value in (
        ("sticker", "sticker"),
        ("photo", "photo"),
        ("video", "video"),
        ("voice", "voice"),
        ("audio", "audio"),
        ("document", "document"),
        ("location", "location"),
    ):
        if getattr(message, attr, None):
            return value
    return "text"


def _media_prefix(message: Any, message_type: str) -> str:
    if message_type == "sticker":
        sticker = getattr(message, "sticker", None)
        emoji = getattr(sticker, "emoji", None) if sticker is not None else None
        return f"[sticker {emoji}]" if emoji else "[sticker]"
    if message_type == "photo":
        return "[photo]"
    if message_type == "video":
        return "[video]"
    if message_type == "voice":
        return "[voice]"
    if message_type == "audio":
        audio = getattr(message, "audio", None)
        name = getattr(audio, "file_name", None) if audio is not None else None
        return f"[audio: {name}]" if name else "[audio]"
    if message_type == "document":
        doc = getattr(message, "document", None)
        name = getattr(doc, "file_name", None) if doc is not None else None
        return f"[document: {name}]" if name else "[document]"
    if message_type == "location":
        location = getattr(message, "location", None)
        lat = getattr(location, "latitude", None) if location is not None else None
        lon = getattr(location, "longitude", None) if location is not None else None
        if lat is not None and lon is not None:
            return f"[location {lat}, {lon}]"
        return "[location]"
    return ""


def _message_text(message: Any, msg_type: Any) -> str:
    message_type = _message_type_value(message, msg_type)
    text = str(getattr(message, "text", None) or getattr(message, "caption", None) or "")
    prefix = _media_prefix(message, message_type)
    if prefix and text.strip():
        return f"{prefix} {text}"
    return text or prefix


def _reply_to_message_id(message: Any, event: Any = None) -> str | None:
    if event is not None:
        value = getattr(event, "reply_to_message_id", None)
        if value is not None:
            return str(value)
    reply = getattr(message, "reply_to_message", None)
    reply_id = getattr(reply, "message_id", None) if reply is not None else None
    return str(reply_id) if reply_id is not None else None


def _is_reply_to_bot(message: Any, adapter: Any) -> bool:
    try:
        return bool(adapter._is_reply_to_bot(message))
    except Exception:
        return False


def _mentions_bot(message: Any, adapter: Any) -> bool:
    try:
        return bool(adapter._message_mentions_bot(message))
    except Exception:
        return False


def _was_trigger(message: Any, event: Any, msg_type: Any, adapter: Any) -> bool:
    text = str(getattr(message, "text", None) or getattr(message, "caption", None) or "")
    lowered = text.casefold()
    if _is_reply_to_bot(message, adapter):
        return True
    if _mentions_bot(message, adapter):
        return True
    if "/ai" in lowered:
        return True
    return _message_type_value(message, msg_type) == "command"


def _upsert_message(
    conn: sqlite3.Connection,
    *,
    chat_id: str,
    telegram_message_id: str,
    sender_user_id: str | None,
    sender_name_snapshot: str | None,
    sent_at: str,
    text: str,
    message_type: str,
    reply_to_message_id: str | None,
    is_bot_message: bool,
    is_neuromancer_message: bool,
    was_trigger: bool,
    raw_json: str,
) -> None:
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
          text = excluded.text,
          message_type = excluded.message_type,
          reply_to_message_id = COALESCE(excluded.reply_to_message_id, messages_raw.reply_to_message_id),
          is_bot_message = excluded.is_bot_message,
          is_neuromancer_message = excluded.is_neuromancer_message,
          was_trigger = CASE
            WHEN messages_raw.was_trigger = 1 OR excluded.was_trigger = 1 THEN 1
            ELSE 0
          END,
          raw_json = excluded.raw_json
        """,
        (
            chat_id,
            telegram_message_id,
            sender_user_id,
            sender_name_snapshot,
            sent_at,
            text,
            message_type,
            reply_to_message_id,
            int(is_bot_message),
            int(is_neuromancer_message),
            int(was_trigger),
            raw_json,
        ),
    )


def _record_inbound_message(
    *,
    message: Any,
    msg_type: Any,
    adapter: Any,
    event: Any = None,
    update_id: int | None = None,
) -> None:
    telegram_message_id = getattr(message, "message_id", None)
    if telegram_message_id is None:
        return

    with _connect() as conn:
        chat_id = _chat_id_from_message(message) or _chat_id_from_event(event)
        if not _target_chat_matches(conn, chat_id):
            return
        chat_id = str(chat_id)

        sender_user_id, sender_name, username, is_bot = _user_from_message(message)
        is_neuromancer = _is_neuromancer_user(sender_user_id, adapter)
        if sender_user_id:
            _ensure_member(
                conn,
                user_id=sender_user_id,
                chat_id=chat_id,
                display_name=sender_name,
                username=username,
                is_bot=is_bot or is_neuromancer,
                is_neuromancer=is_neuromancer,
            )

        raw_payload = {
            "platform": "telegram",
            "direction": "inbound",
            "update_id": update_id,
            "message": _raw_message_payload(message),
        }
        message_type = _message_type_value(message, msg_type)
        _upsert_message(
            conn,
            chat_id=chat_id,
            telegram_message_id=str(telegram_message_id),
            sender_user_id=sender_user_id,
            sender_name_snapshot=sender_name,
            sent_at=_to_iso(getattr(message, "date", None)),
            text=_message_text(message, msg_type),
            message_type=message_type,
            reply_to_message_id=_reply_to_message_id(message, event),
            is_bot_message=is_bot or is_neuromancer,
            is_neuromancer_message=is_neuromancer,
            was_trigger=_was_trigger(message, event, msg_type, adapter),
            raw_json=_json_dumps(raw_payload),
        )
        _refresh_member_stats(conn, chat_id, sender_user_id)
        conn.commit()


def _split_for_message_ids(content: str, ids: list[str]) -> list[str]:
    if len(ids) <= 1:
        return [content]
    if not content:
        return ["" for _ in ids]
    chunk_size = max(1, math.ceil(len(content) / len(ids)))
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    if len(chunks) < len(ids):
        chunks.extend([""] * (len(ids) - len(chunks)))
    return chunks[: len(ids)]


def _result_message_ids(result: Any) -> list[str]:
    ids: list[str] = []
    raw = getattr(result, "raw_response", None) or {}
    if isinstance(raw, dict):
        for value in raw.get("message_ids") or ():
            text = str(value)
            if text and text not in ids:
                ids.append(text)
    for value in (getattr(result, "message_id", None),):
        if value is None:
            continue
        text = str(value)
        if text and text not in ids:
            ids.append(text)
    for value in getattr(result, "continuation_message_ids", None) or ():
        text = str(value)
        if text and text not in ids:
            ids.append(text)
    return ids


def _record_outgoing_messages(
    *,
    adapter: Any,
    chat_id: Any,
    content: str,
    message_ids: Iterable[str],
    metadata: dict[str, Any] | None,
    reply_to: Any = None,
    edited: bool = False,
) -> None:
    ids = [str(mid) for mid in message_ids if mid is not None and str(mid)]
    if not ids:
        return

    with _connect() as conn:
        if not _target_chat_matches(conn, chat_id):
            return
        chat_id = str(chat_id)
        user_id, display_name, username = _bot_identity(adapter)
        _ensure_member(
            conn,
            user_id=user_id,
            chat_id=chat_id,
            display_name=display_name,
            username=username,
            is_bot=True,
            is_neuromancer=True,
        )
        chunks = _split_for_message_ids(content, ids)
        for message_id, chunk in zip(ids, chunks):
            raw_payload = {
                "platform": "telegram",
                "direction": "outgoing",
                "source": "hermes_gateway",
                "edited": edited,
                "chat_id": chat_id,
                "message_id": message_id,
                "reply_to_message_id": str(reply_to) if reply_to is not None else None,
                "metadata": metadata or {},
                "text": chunk,
                "recorded_at": _utc_now(),
            }
            _upsert_message(
                conn,
                chat_id=chat_id,
                telegram_message_id=message_id,
                sender_user_id=user_id,
                sender_name_snapshot=display_name,
                sent_at=_utc_now(),
                text=chunk,
                message_type="text",
                reply_to_message_id=str(reply_to) if reply_to is not None else None,
                is_bot_message=True,
                is_neuromancer_message=True,
                was_trigger=False,
                raw_json=_json_dumps(raw_payload),
            )
        _refresh_member_stats(conn, chat_id, user_id)
        conn.commit()


def _on_pre_gateway_dispatch(event: Any = None, **_: Any) -> None:
    """Safety net for non-Telegram adapters or patched-method misses."""
    if event is None or getattr(getattr(event, "source", None), "platform", None) is None:
        return
    source = getattr(event, "source", None)
    platform = getattr(getattr(source, "platform", None), "value", None) or str(getattr(source, "platform", ""))
    if platform != "telegram":
        return
    raw_message = getattr(event, "raw_message", None)
    if raw_message is None:
        return
    try:
        _record_inbound_message(
            message=raw_message,
            msg_type=getattr(event, "message_type", None),
            adapter=None,
            event=event,
            update_id=getattr(event, "platform_update_id", None),
        )
    except Exception as exc:
        LOGGER.debug("pre_gateway_dispatch recording failed: %s", exc, exc_info=True)


def _patch_telegram_adapter() -> None:
    try:
        from plugins.platforms.telegram.adapter import TelegramAdapter
    except Exception as exc:
        LOGGER.debug("Telegram adapter unavailable for memory patch: %s", exc, exc_info=True)
        return

    if getattr(TelegramAdapter, "_neuromancer_memory_patched", False):
        return

    original_build = TelegramAdapter._build_message_event
    original_command = TelegramAdapter._handle_command
    original_send = TelegramAdapter.send
    original_edit = TelegramAdapter.edit_message

    def patched_build(self, message, msg_type, update_id=None):
        event = original_build(self, message, msg_type, update_id=update_id)
        try:
            _record_inbound_message(
                message=message,
                msg_type=msg_type,
                adapter=self,
                event=event,
                update_id=update_id,
            )
        except Exception as exc:
            LOGGER.debug("Telegram inbound memory recording failed: %s", exc, exc_info=True)
        return event

    async def patched_command(self, update, context):
        msg = self._effective_update_message(update)
        if msg is not None:
            try:
                _record_inbound_message(
                    message=msg,
                    msg_type="command",
                    adapter=self,
                    event=None,
                    update_id=getattr(update, "update_id", None),
                )
            except Exception as exc:
                LOGGER.debug("Telegram command memory recording failed: %s", exc, exc_info=True)
        return await original_command(self, update, context)

    async def patched_send(self, chat_id, content, reply_to=None, metadata=None):
        result = await original_send(self, chat_id, content, reply_to=reply_to, metadata=metadata)
        if getattr(result, "success", False):
            try:
                _record_outgoing_messages(
                    adapter=self,
                    chat_id=chat_id,
                    content=content,
                    message_ids=_result_message_ids(result),
                    metadata=metadata,
                    reply_to=reply_to,
                    edited=False,
                )
            except Exception as exc:
                LOGGER.debug("Telegram outgoing memory recording failed: %s", exc, exc_info=True)
        return result

    async def patched_edit(self, chat_id, message_id, content, *, finalize=False, metadata=None):
        result = await original_edit(
            self,
            chat_id,
            message_id,
            content,
            finalize=finalize,
            metadata=metadata,
        )
        if getattr(result, "success", False):
            try:
                ids = _result_message_ids(result) or [str(message_id)]
                _record_outgoing_messages(
                    adapter=self,
                    chat_id=chat_id,
                    content=content,
                    message_ids=ids,
                    metadata=metadata,
                    reply_to=None,
                    edited=True,
                )
            except Exception as exc:
                LOGGER.debug("Telegram outgoing edit memory recording failed: %s", exc, exc_info=True)
        return result

    TelegramAdapter._neuromancer_memory_original_build_message_event = original_build
    TelegramAdapter._neuromancer_memory_original_handle_command = original_command
    TelegramAdapter._neuromancer_memory_original_send = original_send
    TelegramAdapter._neuromancer_memory_original_edit_message = original_edit
    TelegramAdapter._build_message_event = patched_build
    TelegramAdapter._handle_command = patched_command
    TelegramAdapter.send = patched_send
    TelegramAdapter.edit_message = patched_edit
    TelegramAdapter._neuromancer_memory_patched = True


def register(ctx) -> None:
    ctx.register_hook("pre_gateway_dispatch", _on_pre_gateway_dispatch)
    _patch_telegram_adapter()

