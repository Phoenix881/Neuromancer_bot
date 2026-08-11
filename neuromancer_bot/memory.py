"""SQLite memory access for the standalone Telegram runtime."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


LEGACY_NEUROMANCER_USER_ID = "8967384440"


@dataclass(frozen=True)
class GroupProfile:
    chat_id: str
    profile: str


@dataclass(frozen=True)
class MemberInfo:
    user_id: str
    primary_name: str
    username: str | None
    is_bot: bool
    profile: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class MessageAttachment:
    media_type: str
    local_path: str | None
    width: int | None
    height: int | None
    file_size: int | None
    vision_status: str
    vision_summary: str | None
    vision_ocr: str | None


@dataclass(frozen=True)
class RecentMessage:
    telegram_message_id: str
    sender_user_id: str | None
    sender_name_snapshot: str | None
    sent_at: str
    text: str
    message_type: str
    reply_to_message_id: str | None
    is_bot_message: bool
    is_neuromancer_message: bool
    was_trigger: bool
    attachments: tuple[MessageAttachment, ...] = ()


@dataclass(frozen=True)
class MemoryContext:
    group: GroupProfile
    members: tuple[MemberInfo, ...]
    recent_messages: tuple[RecentMessage, ...]
    active_member: MemberInfo | None


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def to_iso(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if value:
        try:
            return datetime.fromisoformat(str(value)).isoformat()
        except ValueError:
            return str(value)
    return utc_now()


def json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        return to_iso(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return value.to_dict()
        except Exception:
            pass
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=json_default)


def normalized_alias(value: str) -> str:
    return value.strip().casefold()


class SQLiteMemory:
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn

    def ensure_media_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS media_sends (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              asset_id TEXT NOT NULL,
              asset_type TEXT NOT NULL,
              chat_id TEXT NOT NULL,
              telegram_message_id TEXT NOT NULL,
              sent_at TEXT NOT NULL,
              reply_to_message_id TEXT,
              caption TEXT,
              raw_json TEXT,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(chat_id, telegram_message_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_media_sends_asset_sent
            ON media_sends(asset_id, sent_at DESC, id DESC)
            """
        )

    def ensure_attachment_tables(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS message_attachments (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              message_raw_id INTEGER NOT NULL REFERENCES messages_raw(id) ON DELETE CASCADE,
              telegram_file_id TEXT NOT NULL,
              telegram_file_unique_id TEXT,
              media_type TEXT NOT NULL,
              local_path TEXT,
              mime_type TEXT,
              width INTEGER,
              height INTEGER,
              file_size INTEGER,
              sha256 TEXT,
              vision_status TEXT NOT NULL DEFAULT 'pending',
              vision_summary TEXT,
              vision_ocr TEXT,
              vision_json TEXT,
              error TEXT,
              created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(message_raw_id, telegram_file_id)
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_message_attachments_message
            ON message_attachments(message_raw_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_message_attachments_vision_status
            ON message_attachments(vision_status, updated_at DESC)
            """
        )

    def target_chat_id(self) -> str:
        with self.connect() as conn:
            row = conn.execute("SELECT chat_id FROM group_profile WHERE id = 1").fetchone()
        if not row or row[0] is None:
            raise RuntimeError("group_profile row with id=1 is missing")
        return str(row[0])

    def ensure_member(
        self,
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
                "recorded by the standalone Python Telegram runtime."
            )
        else:
            profile = f"Live Telegram member first seen as {primary}."

        # On conflict, curated identity/personality text remains manually owned.
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

        for alias in aliases_for(display_name, username):
            conn.execute(
                """
                INSERT OR IGNORE INTO member_aliases (
                  user_id, alias, normalized_alias, source
                )
                VALUES (?, ?, ?, 'live')
                """,
                (user_id, alias, normalized_alias(alias)),
            )

    def refresh_member_stats(
        self, conn: sqlite3.Connection, *, chat_id: str, user_id: str | None
    ) -> None:
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

    def upsert_message(
        self,
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
              sent_at = excluded.sent_at,
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

    def message_raw_id(
        self,
        conn: sqlite3.Connection,
        *,
        chat_id: str,
        telegram_message_id: str,
    ) -> int | None:
        row = conn.execute(
            """
            SELECT id
            FROM messages_raw
            WHERE chat_id = ? AND telegram_message_id = ?
            """,
            (chat_id, telegram_message_id),
        ).fetchone()
        return int(row[0]) if row else None

    def record_incoming_message(
        self,
        message: Any,
        *,
        bot_user_id: str | None,
        was_trigger: bool,
    ) -> int | None:
        telegram_message_id = getattr(message, "message_id", None)
        if telegram_message_id is None:
            return None

        chat_id = str(getattr(getattr(message, "chat", None), "id", ""))
        if not chat_id:
            return None

        sender_user_id, sender_name, username, telegram_says_bot = user_from_message(message)
        is_neuromancer = is_neuromancer_user(sender_user_id, bot_user_id)
        is_bot = telegram_says_bot or is_neuromancer

        with self.connect() as conn:
            if sender_user_id:
                self.ensure_member(
                    conn,
                    user_id=sender_user_id,
                    chat_id=chat_id,
                    display_name=sender_name,
                    username=username,
                    is_bot=is_bot,
                    is_neuromancer=is_neuromancer,
                )

            message_type = message_type_value(message)
            raw_payload = {
                "platform": "telegram",
                "direction": "inbound",
                "message": raw_message_payload(message),
            }
            self.upsert_message(
                conn,
                chat_id=chat_id,
                telegram_message_id=str(telegram_message_id),
                sender_user_id=sender_user_id,
                sender_name_snapshot=sender_name,
                sent_at=to_iso(getattr(message, "date", None)),
                text=message_text(message, message_type),
                message_type=message_type,
                reply_to_message_id=reply_to_message_id(message),
                is_bot_message=is_bot,
                is_neuromancer_message=is_neuromancer,
                was_trigger=was_trigger,
                raw_json=json_dumps(raw_payload),
            )
            self.refresh_member_stats(conn, chat_id=chat_id, user_id=sender_user_id)
            message_row_id = self.message_raw_id(
                conn,
                chat_id=chat_id,
                telegram_message_id=str(telegram_message_id),
            )
            conn.commit()
            return message_row_id

    def record_attachment(
        self,
        *,
        message_raw_id: int,
        telegram_file_id: str,
        telegram_file_unique_id: str | None,
        media_type: str,
        local_path: str | None,
        mime_type: str | None,
        width: int | None,
        height: int | None,
        file_size: int | None,
        sha256: str | None,
        vision_status: str = "pending",
    ) -> int:
        with self.connect() as conn:
            self.ensure_attachment_tables(conn)
            conn.execute(
                """
                INSERT INTO message_attachments (
                  message_raw_id,
                  telegram_file_id,
                  telegram_file_unique_id,
                  media_type,
                  local_path,
                  mime_type,
                  width,
                  height,
                  file_size,
                  sha256,
                  vision_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_raw_id, telegram_file_id) DO UPDATE SET
                  telegram_file_unique_id = excluded.telegram_file_unique_id,
                  media_type = excluded.media_type,
                  local_path = excluded.local_path,
                  mime_type = excluded.mime_type,
                  width = excluded.width,
                  height = excluded.height,
                  file_size = excluded.file_size,
                  sha256 = excluded.sha256,
                  vision_status = excluded.vision_status,
                  updated_at = CURRENT_TIMESTAMP
                """,
                (
                    message_raw_id,
                    telegram_file_id,
                    telegram_file_unique_id,
                    media_type,
                    local_path,
                    mime_type,
                    width,
                    height,
                    file_size,
                    sha256,
                    vision_status,
                ),
            )
            row = conn.execute(
                """
                SELECT id
                FROM message_attachments
                WHERE message_raw_id = ? AND telegram_file_id = ?
                """,
                (message_raw_id, telegram_file_id),
            ).fetchone()
            conn.commit()
            if not row:
                raise RuntimeError("Attachment upsert did not return a row id")
            return int(row[0])

    def update_attachment_vision(
        self,
        *,
        attachment_id: int,
        vision_status: str,
        vision_summary: str | None = None,
        vision_ocr: str | None = None,
        vision_json: str | None = None,
        error: str | None = None,
    ) -> None:
        with self.connect() as conn:
            self.ensure_attachment_tables(conn)
            conn.execute(
                """
                UPDATE message_attachments
                SET
                  vision_status = ?,
                  vision_summary = ?,
                  vision_ocr = ?,
                  vision_json = ?,
                  error = ?,
                  updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    vision_status,
                    vision_summary,
                    vision_ocr,
                    vision_json,
                    error,
                    attachment_id,
                ),
            )
            conn.commit()

    def record_outgoing_message(
        self,
        message: Any,
        *,
        bot_user_id: str,
        bot_display_name: str,
        bot_username: str | None,
        text: str,
        reply_to_message_id_value: str | None,
    ) -> None:
        telegram_message_id = getattr(message, "message_id", None)
        chat_id = str(getattr(getattr(message, "chat", None), "id", ""))
        if telegram_message_id is None or not chat_id:
            return

        with self.connect() as conn:
            self.ensure_member(
                conn,
                user_id=bot_user_id,
                chat_id=chat_id,
                display_name=bot_display_name,
                username=bot_username,
                is_bot=True,
                is_neuromancer=True,
            )
            raw_payload = {
                "platform": "telegram",
                "direction": "outgoing",
                "source": "neuromancer_bot",
                "message": raw_message_payload(message),
                "reply_to_message_id": reply_to_message_id_value,
                "text": text,
                "recorded_at": utc_now(),
            }
            self.upsert_message(
                conn,
                chat_id=chat_id,
                telegram_message_id=str(telegram_message_id),
                sender_user_id=bot_user_id,
                sender_name_snapshot=bot_display_name,
                sent_at=to_iso(getattr(message, "date", None)),
                text=text,
                message_type="text",
                reply_to_message_id=reply_to_message_id_value,
                is_bot_message=True,
                is_neuromancer_message=True,
                was_trigger=False,
                raw_json=json_dumps(raw_payload),
            )
            self.refresh_member_stats(conn, chat_id=chat_id, user_id=bot_user_id)
            conn.commit()

    def record_outgoing_media(
        self,
        message: Any,
        *,
        bot_user_id: str,
        bot_display_name: str,
        bot_username: str | None,
        asset_id: str,
        asset_type: str,
        caption: str | None,
        reply_to_message_id_value: str | None,
    ) -> None:
        telegram_message_id = getattr(message, "message_id", None)
        chat_id = str(getattr(getattr(message, "chat", None), "id", ""))
        if telegram_message_id is None or not chat_id:
            return

        caption_text = (caption or "").strip()
        transcript_text = f"[{asset_type}:{asset_id}]"
        if caption_text:
            transcript_text = f"{transcript_text} {caption_text}"

        with self.connect() as conn:
            self.ensure_media_tables(conn)
            self.ensure_member(
                conn,
                user_id=bot_user_id,
                chat_id=chat_id,
                display_name=bot_display_name,
                username=bot_username,
                is_bot=True,
                is_neuromancer=True,
            )
            raw_payload = {
                "platform": "telegram",
                "direction": "outgoing",
                "source": "neuromancer_bot",
                "message": raw_message_payload(message),
                "media_asset_id": asset_id,
                "media_asset_type": asset_type,
                "reply_to_message_id": reply_to_message_id_value,
                "caption": caption_text,
                "recorded_at": utc_now(),
            }
            raw_json = json_dumps(raw_payload)
            sent_at = to_iso(getattr(message, "date", None))
            self.upsert_message(
                conn,
                chat_id=chat_id,
                telegram_message_id=str(telegram_message_id),
                sender_user_id=bot_user_id,
                sender_name_snapshot=bot_display_name,
                sent_at=sent_at,
                text=transcript_text,
                message_type=asset_type,
                reply_to_message_id=reply_to_message_id_value,
                is_bot_message=True,
                is_neuromancer_message=True,
                was_trigger=False,
                raw_json=raw_json,
            )
            conn.execute(
                """
                INSERT INTO media_sends (
                  asset_id,
                  asset_type,
                  chat_id,
                  telegram_message_id,
                  sent_at,
                  reply_to_message_id,
                  caption,
                  raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, telegram_message_id) DO UPDATE SET
                  asset_id = excluded.asset_id,
                  asset_type = excluded.asset_type,
                  sent_at = excluded.sent_at,
                  reply_to_message_id = excluded.reply_to_message_id,
                  caption = excluded.caption,
                  raw_json = excluded.raw_json
                """,
                (
                    asset_id,
                    asset_type,
                    chat_id,
                    str(telegram_message_id),
                    sent_at,
                    reply_to_message_id_value,
                    caption_text,
                    raw_json,
                ),
            )
            self.refresh_member_stats(conn, chat_id=chat_id, user_id=bot_user_id)
            conn.commit()

    def load_context(
        self,
        *,
        chat_id: str,
        active_user_id: str | None,
        recent_limit: int,
    ) -> MemoryContext:
        with self.connect() as conn:
            group_row = conn.execute(
                "SELECT chat_id, profile FROM group_profile WHERE chat_id = ?",
                (chat_id,),
            ).fetchone()
            if not group_row:
                raise RuntimeError(f"No group_profile found for chat_id {chat_id}")

            members = tuple(load_members(conn, chat_id))
            active = next((member for member in members if member.user_id == active_user_id), None)

            rows = conn.execute(
                """
                SELECT
                  id,
                  telegram_message_id,
                  sender_user_id,
                  sender_name_snapshot,
                  sent_at,
                  COALESCE(text, ''),
                  message_type,
                  reply_to_message_id,
                  is_bot_message,
                  is_neuromancer_message,
                  was_trigger
                FROM messages_raw
                WHERE chat_id = ?
                ORDER BY sent_at DESC, id DESC
                LIMIT ?
                """,
                (chat_id, recent_limit),
            ).fetchall()
            attachments_by_message = load_recent_attachments(conn, [int(row[0]) for row in rows])

        recent = tuple(
            RecentMessage(
                telegram_message_id=str(row[1]),
                sender_user_id=str(row[2]) if row[2] is not None else None,
                sender_name_snapshot=str(row[3]) if row[3] is not None else None,
                sent_at=str(row[4]),
                text=str(row[5] or ""),
                message_type=str(row[6] or "text"),
                reply_to_message_id=str(row[7]) if row[7] is not None else None,
                is_bot_message=bool(row[8]),
                is_neuromancer_message=bool(row[9]),
                was_trigger=bool(row[10]),
                attachments=attachments_by_message.get(int(row[0]), ()),
            )
            for row in reversed(rows)
        )
        return MemoryContext(
            group=GroupProfile(chat_id=str(group_row[0]), profile=str(group_row[1])),
            members=members,
            recent_messages=recent,
            active_member=active,
        )


def load_members(conn: sqlite3.Connection, chat_id: str) -> Iterable[MemberInfo]:
    rows = conn.execute(
        """
        SELECT user_id, primary_name, username, is_bot, profile
        FROM members
        WHERE chat_id = ?
        ORDER BY is_bot ASC, lower(primary_name) ASC
        """,
        (chat_id,),
    ).fetchall()
    for row in rows:
        user_id = str(row[0])
        alias_rows = conn.execute(
            """
            SELECT alias
            FROM member_aliases
            WHERE user_id = ?
            ORDER BY source = 'manual' DESC, created_at ASC, alias ASC
            """,
            (user_id,),
        ).fetchall()
        yield MemberInfo(
            user_id=user_id,
            primary_name=str(row[1]),
            username=str(row[2]) if row[2] is not None else None,
            is_bot=bool(row[3]),
            profile=str(row[4]),
            aliases=tuple(str(alias_row[0]) for alias_row in alias_rows),
        )


def load_recent_attachments(
    conn: sqlite3.Connection,
    message_raw_ids: list[int],
) -> dict[int, tuple[MessageAttachment, ...]]:
    if not message_raw_ids:
        return {}
    placeholders = ",".join("?" for _ in message_raw_ids)
    try:
        rows = conn.execute(
            f"""
            SELECT
              message_raw_id,
              media_type,
              local_path,
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
            tuple(message_raw_ids),
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    grouped: dict[int, list[MessageAttachment]] = {}
    for row in rows:
        grouped.setdefault(int(row[0]), []).append(
            MessageAttachment(
                media_type=str(row[1]),
                local_path=str(row[2]) if row[2] is not None else None,
                width=int(row[3]) if row[3] is not None else None,
                height=int(row[4]) if row[4] is not None else None,
                file_size=int(row[5]) if row[5] is not None else None,
                vision_status=str(row[6]),
                vision_summary=str(row[7]) if row[7] is not None else None,
                vision_ocr=str(row[8]) if row[8] is not None else None,
            )
        )
    return {message_id: tuple(items) for message_id, items in grouped.items()}


def raw_message_payload(message: Any) -> dict[str, Any]:
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


def display_name_from_user(user: Any) -> str:
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


def user_from_message(message: Any) -> tuple[str | None, str | None, str | None, bool]:
    user = getattr(message, "from_user", None)
    if user is None:
        return None, None, None, False
    user_id = getattr(user, "id", None)
    username = getattr(user, "username", None)
    display_name = display_name_from_user(user)
    return (
        str(user_id) if user_id is not None else None,
        display_name or None,
        str(username) if username else None,
        bool(getattr(user, "is_bot", False)),
    )


def aliases_for(display_name: str | None, username: str | None) -> list[str]:
    aliases: list[str] = []
    for value in (display_name, username, f"@{username}" if username else None):
        text = str(value).strip() if value else ""
        if text and text not in aliases:
            aliases.append(text)
    return aliases


def is_neuromancer_user(user_id: str | None, bot_user_id: str | None) -> bool:
    if not user_id:
        return False
    return user_id == LEGACY_NEUROMANCER_USER_ID or (
        bot_user_id is not None and str(bot_user_id) == user_id
    )


def reply_to_message_id(message: Any) -> str | None:
    reply = getattr(message, "reply_to_message", None)
    reply_id = getattr(reply, "message_id", None) if reply is not None else None
    return str(reply_id) if reply_id is not None else None


def message_type_value(message: Any) -> str:
    checks = (
        ("text", "text"),
        ("sticker", "sticker"),
        ("photo", "photo"),
        ("video", "video"),
        ("animation", "animation"),
        ("voice", "voice"),
        ("audio", "audio"),
        ("document", "document"),
        ("video_note", "video_note"),
        ("location", "location"),
        ("poll", "poll"),
        ("contact", "contact"),
        ("new_chat_members", "service:new_chat_members"),
        ("left_chat_member", "service:left_chat_member"),
        ("pinned_message", "service:pinned_message"),
    )
    for attr, value in checks:
        if getattr(message, attr, None):
            return value
    return "service"


def media_prefix(message: Any, message_type: str) -> str:
    if message_type == "sticker":
        sticker = getattr(message, "sticker", None)
        emoji = getattr(sticker, "emoji", None) if sticker is not None else None
        return f"[sticker {emoji}]" if emoji else "[sticker]"
    if message_type in {"photo", "video", "animation", "voice", "video_note", "poll", "contact"}:
        return f"[{message_type}]"
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
    if message_type == "service:new_chat_members":
        members = getattr(message, "new_chat_members", None) or []
        names = ", ".join(display_name_from_user(member) for member in members)
        return f"[service] new_chat_members: {names}" if names else "[service] new_chat_members"
    if message_type == "service:left_chat_member":
        member = getattr(message, "left_chat_member", None)
        name = display_name_from_user(member) if member is not None else ""
        return f"[service] left_chat_member: {name}" if name else "[service] left_chat_member"
    if message_type.startswith("service"):
        return f"[{message_type}]"
    return ""


def message_text(message: Any, message_type: str | None = None) -> str:
    kind = message_type or message_type_value(message)
    text = str(getattr(message, "text", None) or getattr(message, "caption", None) or "")
    prefix = media_prefix(message, kind)
    if prefix and text.strip():
        return f"{prefix} {text}"
    return text or prefix


def clamp_text(value: str, limit: int = 900) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)].rstrip() + "..."


def split_text_for_telegram(text: str, limit: int) -> list[str]:
    clean = text.strip()
    if not clean:
        return []
    if len(clean) <= limit:
        return [clean]

    chunks: list[str] = []
    remaining = clean
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < math.floor(limit * 0.55):
            cut = remaining.rfind(" ", 0, limit)
        if cut < math.floor(limit * 0.55):
            cut = limit
        chunk = remaining[:cut].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks
