"""Hermes-backed response generation."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass

from .config import Settings
from .media_library import MediaLibrary
from .memory import MemoryContext, RecentMessage, clamp_text


class HermesError(RuntimeError):
    pass


@dataclass(frozen=True)
class TriggerMessage:
    chat_id: str
    message_id: str
    sender_user_id: str | None
    sender_name: str | None
    sent_at: str
    text: str
    clean_text: str
    reply_to_message_id: str | None
    was_from_bot: bool


@dataclass(frozen=True)
class AgentReply:
    reply_text: str
    media_id: str | None = None
    delivery: str = "reply"


class HermesAgent:
    def __init__(self, settings: Settings, media_library: MediaLibrary):
        self.settings = settings
        self.media_library = media_library

    async def generate_reply(self, trigger: TriggerMessage, context: MemoryContext) -> AgentReply:
        prompt = build_prompt(trigger, context, self.media_library)
        command = self._command(prompt)
        env = os.environ.copy()
        env["HERMES_HOME"] = str(self.settings.hermes_home)
        env.setdefault("HERMES_ACCEPT_HOOKS", "1")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self.settings.hermes_home),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise HermesError(f"Hermes command not found: {self.settings.hermes_command}") from exc

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.settings.hermes_timeout_seconds
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise HermesError(
                f"Hermes timed out after {self.settings.hermes_timeout_seconds:.0f}s"
            ) from exc

        out = stdout.decode("utf-8", errors="replace").strip()
        err = stderr.decode("utf-8", errors="replace").strip()
        if process.returncode != 0:
            detail = err or out or f"exit code {process.returncode}"
            raise HermesError(detail)
        if not out:
            raise HermesError("Hermes returned an empty reply")
        return parse_agent_reply(out, self.media_library)

    def _command(self, prompt: str) -> list[str]:
        command = [self.settings.hermes_command, "--accept-hooks"]
        if self.settings.hermes_provider:
            command.extend(["--provider", self.settings.hermes_provider])
        if self.settings.hermes_model:
            command.extend(["--model", self.settings.hermes_model])
        if self.settings.hermes_reasoning:
            command.extend(["--reasoning", self.settings.hermes_reasoning])
        if self.settings.hermes_toolsets:
            command.extend(["--toolsets", self.settings.hermes_toolsets])
        command.extend(["-z", prompt])
        return command


def build_prompt(
    trigger: TriggerMessage,
    context: MemoryContext,
    media_library: MediaLibrary,
) -> str:
    active = format_active_member(context)
    members = "\n".join(format_member(member) for member in context.members)
    recent = "\n".join(format_recent_message(message) for message in context.recent_messages)

    return f"""SOUL.md is part of your identity in this Hermes profile. Follow it.

Runtime boundary:
- The standalone Python Telegram bot already wrote the incoming message into SQLite.
- Incoming voice/audio/video-note messages are transcribed by Python before you are called when transcription succeeds.
- In recent context and database search results, `[audio transcript] ...` and `[video-note transcript] ...` are real transcripts. Treat them as message text.
- Do not say you lack STT/audio/transcript access when a transcript is present in the prompt or database search results.
- Do not use tools to mutate neuromancer_memory.db or the group/member profile tables.
- Group and friend descriptions are manually owned by Karim.
- Use tools when they genuinely help answer the user.
- Return only a valid JSON object. No transcript, no code fence wrapper.
- JSON schema: {{"reply_text":"telegram reply text","media_id":null,"delivery":"reply"}}
- If a fixed media reaction fits unusually well, set media_id to one listed below.
- If no media is needed, set media_id to null.
- Do not invent media ids or file paths.
- Keep media rare and intentional; these are sharp reaction images, not decoration.
- delivery must be "reply" or "standalone".
- Use delivery="reply" when directly answering the triggering message.
- Use delivery="standalone" for ambient group-chat remarks, standalone reactions, or when attaching the reply visually to the trigger would feel awkward.
- Default to delivery="reply" unless standalone clearly fits better.
- Default to delivery="standalone" if asked to say/send something that is meant for other person, in that case tag them with their username

Available fixed media reactions:
{media_library.prompt_summary()}

Group:
chat_id: {context.group.chat_id}
profile: {context.group.profile}

Known members:
{members or "(none)"}

Active speaker:
{active}

Recent chat context, oldest to newest:
{recent or "(none)"}

Triggered message:
message_id: {trigger.message_id}
sent_at: {trigger.sent_at}
from: {trigger.sender_name or trigger.sender_user_id or "unknown"} ({trigger.sender_user_id or "unknown"})
reply_to_message_id: {trigger.reply_to_message_id or "none"}
text: {trigger.text}
clean_text: {trigger.clean_text or trigger.text}

Reply as Neuromancer now."""


def parse_agent_reply(raw: str, media_library: MediaLibrary) -> AgentReply:
    data = parse_json_object(raw)
    if not isinstance(data, dict):
        return AgentReply(reply_text=raw.strip(), media_id=None, delivery="reply")

    reply_text = data.get("reply_text", data.get("reply", ""))
    if reply_text is None:
        reply_text = ""
    reply_text = str(reply_text).strip()

    media_id_raw = data.get("media_id")
    media_id = str(media_id_raw).strip() if media_id_raw not in (None, "", False) else None
    if media_id in {"null", "None", "none"}:
        media_id = None
    if media_id and not media_library.has(media_id):
        media_id = None

    delivery = str(data.get("delivery", "reply")).strip().casefold()
    if delivery not in {"reply", "standalone"}:
        delivery = "reply"

    if not reply_text and not media_id:
        return AgentReply(reply_text=raw.strip(), media_id=None, delivery="reply")
    return AgentReply(reply_text=reply_text, media_id=media_id, delivery=delivery)


def parse_json_object(raw: str) -> object | None:
    text = raw.strip()
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


def format_active_member(context: MemoryContext) -> str:
    if context.active_member is None:
        return "Unknown or not yet profiled."
    return format_member(context.active_member)


def format_member(member) -> str:
    username = f"@{member.username}" if member.username else "no username"
    bot_marker = " bot" if member.is_bot else ""
    aliases = ", ".join(member.aliases) if member.aliases else "none"
    return (
        f"- {member.primary_name}{bot_marker} | user_id={member.user_id} | "
        f"{username} | aliases: {aliases}\n  profile: {member.profile}"
    )


def format_recent_message(message: RecentMessage) -> str:
    sender = message.sender_name_snapshot or message.sender_user_id or "unknown"
    markers: list[str] = []
    if message.is_neuromancer_message:
        markers.append("neuromancer")
    elif message.is_bot_message:
        markers.append("bot")
    if message.was_trigger:
        markers.append("trigger")
    marker_text = f" [{' '.join(markers)}]" if markers else ""
    reply = f" reply_to={message.reply_to_message_id}" if message.reply_to_message_id else ""
    text = clamp_text(message.text or f"[{message.message_type}]", 900)
    base = f"[{message.sent_at}] {sender}{marker_text}{reply}: {text}"
    attachment_lines = [format_attachment(attachment) for attachment in message.attachments]
    attachment_lines = [line for line in attachment_lines if line]
    if attachment_lines:
        return "\n".join([base, *attachment_lines])
    return base


def format_attachment(attachment) -> str:
    size = ""
    if attachment.width and attachment.height:
        size = f" {attachment.width}x{attachment.height}"
    if attachment.vision_status == "complete":
        parts = [f"  vision {attachment.media_type}{size}: {attachment.vision_summary or ''}".rstrip()]
        if attachment.vision_ocr:
            parts.append(f"  ocr: {clamp_text(attachment.vision_ocr, 500)}")
        return "\n".join(parts)
    if attachment.vision_status in {"pending", "running"}:
        return f"  vision {attachment.media_type}{size}: pending"
    if attachment.vision_status in {"failed", "skipped"}:
        return f"  vision {attachment.media_type}{size}: {attachment.vision_status}"
    return ""
