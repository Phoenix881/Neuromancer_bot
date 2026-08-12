"""Read-only Hermes tools for Neuromancer's message database."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import sys
from typing import Any


LOGGER = logging.getLogger(__name__)
PROFILE_ROOT = Path(__file__).resolve().parents[2]
if str(PROFILE_ROOT) not in sys.path:
    sys.path.insert(0, str(PROFILE_ROOT))

from neuromancer_bot.memory_search import lookup_message, recent_user_messages, search_messages


MEMORY_SEARCH_SCHEMA = {
    "name": "neuromancer_memory_search",
    "description": (
        "Read-only search in Neuromancer's Telegram message database. Use for old chat quotes, "
        "exact words/phrases, FTS terms, Telegram message ids, raw DB ids, or recent messages by user id. "
        "Does not modify memory or member/group profiles."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Words, exact phrase, or id to search. Wrap a phrase in quotes for FTS phrase search. "
                    "Use raw:123/db:123 for raw DB id or tg:123 for Telegram message id."
                ),
            },
            "mode": {
                "type": "string",
                "enum": ["auto", "fts", "exact", "id"],
                "description": (
                    "auto detects ids and otherwise uses FTS with exact fallback. "
                    "fts searches tokenized words. exact searches raw substring. id looks up one message."
                ),
                "default": "auto",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum result count. Default 10, hard max 30.",
                "minimum": 1,
                "maximum": 30,
            },
            "sender_user_id": {
                "type": "string",
                "description": "Optional Telegram user id filter.",
            },
            "after": {
                "type": "string",
                "description": "Optional lower sent_at bound, ISO-ish string.",
            },
            "before": {
                "type": "string",
                "description": "Optional upper sent_at bound, ISO-ish string.",
            },
        },
        "required": ["query"],
    },
}


MESSAGE_CONTEXT_SCHEMA = {
    "name": "neuromancer_message_context",
    "description": (
        "Read-only lookup of one message plus nearby messages from Neuromancer's SQLite memory. "
        "Use after neuromancer_memory_search finds a candidate, or when the user gives a Telegram/raw message id."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "message_id": {
                "type": "string",
                "description": "Telegram message id by default. Prefix raw:123/db:123 for raw DB id or tg:123.",
            },
            "id_type": {
                "type": "string",
                "enum": ["auto", "telegram", "raw"],
                "description": "How to interpret message_id. auto tries Telegram id first, then raw DB id.",
                "default": "auto",
            },
            "context_window": {
                "type": "integer",
                "description": "Messages before and after the target. Default 5, hard max 20.",
                "minimum": 0,
                "maximum": 20,
            },
        },
        "required": ["message_id"],
    },
}


USER_MESSAGES_SCHEMA = {
    "name": "neuromancer_user_messages",
    "description": (
        "Read-only lookup of recent messages from one Telegram user id in Neuromancer's message database."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sender_user_id": {
                "type": "string",
                "description": "Telegram user id.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum result count. Default 10, hard max 30.",
                "minimum": 1,
                "maximum": 30,
            },
        },
        "required": ["sender_user_id"],
    },
}


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _handle_memory_search(args: dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    return _json(
        search_messages(
            query=str(args.get("query", "")),
            mode=str(args.get("mode", "auto")),
            limit=args.get("limit", 10),
            sender_user_id=args.get("sender_user_id"),
            after=args.get("after"),
            before=args.get("before"),
        )
    )


def _handle_message_context(args: dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    return _json(
        lookup_message(
            message_id=str(args.get("message_id", "")),
            id_type=str(args.get("id_type", "auto")),
            context_window=args.get("context_window", 5),
        )
    )


def _handle_user_messages(args: dict[str, Any], **kwargs: Any) -> str:
    del kwargs
    return _json(
        recent_user_messages(
            sender_user_id=str(args.get("sender_user_id", "")),
            limit=args.get("limit", 10),
        )
    )


def register(ctx) -> None:
    ctx.register_tool(
        name="neuromancer_memory_search",
        toolset="memory",
        schema=MEMORY_SEARCH_SCHEMA,
        handler=_handle_memory_search,
        description=MEMORY_SEARCH_SCHEMA["description"],
        emoji="DB",
    )
    ctx.register_tool(
        name="neuromancer_message_context",
        toolset="memory",
        schema=MESSAGE_CONTEXT_SCHEMA,
        handler=_handle_message_context,
        description=MESSAGE_CONTEXT_SCHEMA["description"],
        emoji="DB",
    )
    ctx.register_tool(
        name="neuromancer_user_messages",
        toolset="memory",
        schema=USER_MESSAGES_SCHEMA,
        handler=_handle_user_messages,
        description=USER_MESSAGES_SCHEMA["description"],
        emoji="DB",
    )
    LOGGER.info("Registered Neuromancer read-only database search tools")
