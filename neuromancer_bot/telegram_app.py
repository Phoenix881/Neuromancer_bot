"""Telegram polling application for Neuromancer."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from telegram import Message, Update
from telegram.constants import ChatAction
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from .agent import AgentReply, HermesAgent, HermesError, TriggerMessage
from .config import Settings
from .media_library import MediaAsset, MediaLibrary
from .memory import (
    LEGACY_NEUROMANCER_USER_ID,
    SQLiteMemory,
    message_text,
    reply_to_message_id,
    split_text_for_telegram,
    to_iso,
    user_from_message,
)
from .vision import OpenRouterVisionClient, VisionError


LOGGER = logging.getLogger(__name__)
TELEGRAM_CAPTION_LIMIT = 1024


@dataclass(frozen=True)
class IncomingImage:
    file_id: str
    file_unique_id: str | None
    media_type: str
    mime_type: str
    extension: str
    width: int | None
    height: int | None
    file_size: int | None


class NeuromancerTelegramRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.memory = SQLiteMemory(settings.db_path)
        self.media_library = MediaLibrary.load(settings.media_index_path)
        self.agent = HermesAgent(settings, self.media_library)
        self.vision_client = OpenRouterVisionClient(
            model=settings.vision_model,
            base_url=settings.vision_base_url,
            timeout_seconds=settings.vision_timeout_seconds,
        )
        self.reply_lock = asyncio.Lock()
        self.bot_user_id: str | None = None
        self.bot_username: str | None = None
        self.bot_display_name = "Neuromancer"

    def build_application(self) -> Application:
        application = (
            Application.builder()
            .token(self.settings.telegram_bot_token)
            .post_init(self.post_init)
            .build()
        )
        application.add_handler(MessageHandler(filters.ALL, self.handle_message))
        return application

    async def post_init(self, application: Application) -> None:
        me = await application.bot.get_me()
        self.bot_user_id = str(me.id)
        self.bot_username = me.username
        self.bot_display_name = me.full_name or me.username or "Neuromancer"
        LOGGER.info(
            "Neuromancer Telegram runtime ready as %s (%s), target chat %s",
            self.bot_display_name,
            self.bot_user_id,
            self.settings.target_chat_id,
        )

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        if message is None:
            return
        if str(message.chat_id) != self.settings.target_chat_id:
            LOGGER.debug("Ignoring chat %s", message.chat_id)
            return

        was_trigger = self.should_trigger(message)
        message_raw_id = self.memory.record_incoming_message(
            message,
            bot_user_id=self.bot_user_id,
            was_trigger=was_trigger,
        )
        await self.process_incoming_images(message, context, message_raw_id)

        if not was_trigger:
            return
        if self.should_ignore_trigger_from_bot(message):
            LOGGER.info("Stored bot message %s but skipped reply", message.message_id)
            return

        async with self.reply_lock:
            await self.reply_to_trigger(message, context)

    async def process_incoming_images(
        self,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        message_raw_id: int | None,
    ) -> None:
        if message_raw_id is None:
            return

        images = incoming_images_from_message(message)
        if not images:
            return

        for image in images:
            attachment_id: int | None = None
            try:
                path = await self.download_incoming_image(message, context, image)
                sha256 = file_sha256(path)
                attachment_id = self.memory.record_attachment(
                    message_raw_id=message_raw_id,
                    telegram_file_id=image.file_id,
                    telegram_file_unique_id=image.file_unique_id,
                    media_type=image.media_type,
                    local_path=str(path),
                    mime_type=image.mime_type,
                    width=image.width,
                    height=image.height,
                    file_size=image.file_size,
                    sha256=sha256,
                    vision_status="pending",
                )
                await self.run_vision_for_attachment(attachment_id, path)
            except Exception as exc:
                LOGGER.exception(
                    "Failed to process incoming image for Telegram message %s",
                    message.message_id,
                )
                if attachment_id is not None:
                    self.memory.update_attachment_vision(
                        attachment_id=attachment_id,
                        vision_status="failed",
                        error=str(exc),
                    )

    async def download_incoming_image(
        self,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        image: IncomingImage,
    ) -> Path:
        sent_at = message.date
        year = f"{sent_at.year:04d}" if sent_at else "unknown"
        month = f"{sent_at.month:02d}" if sent_at else "unknown"
        directory = self.settings.incoming_media_dir / year / month
        directory.mkdir(parents=True, exist_ok=True)
        unique = sanitize_path_part(image.file_unique_id or image.file_id)
        filename = f"{message.chat_id}_{message.message_id}_{unique}{image.extension}"
        path = directory / filename
        telegram_file = await context.bot.get_file(image.file_id)
        await telegram_file.download_to_drive(custom_path=str(path))
        return path

    async def run_vision_for_attachment(self, attachment_id: int, path: Path) -> None:
        if not self.settings.vision_enabled:
            self.memory.update_attachment_vision(
                attachment_id=attachment_id,
                vision_status="skipped",
                error="vision disabled",
            )
            return
        if self.settings.vision_provider != "openrouter":
            self.memory.update_attachment_vision(
                attachment_id=attachment_id,
                vision_status="skipped",
                error=f"unsupported vision provider: {self.settings.vision_provider}",
            )
            return

        self.memory.update_attachment_vision(attachment_id=attachment_id, vision_status="running")
        try:
            result = await asyncio.to_thread(self.vision_client.analyze_image, path)
        except VisionError as exc:
            LOGGER.warning("Vision analysis failed for %s: %s", path, exc)
            self.memory.update_attachment_vision(
                attachment_id=attachment_id,
                vision_status="failed",
                error=str(exc),
            )
            return

        self.memory.update_attachment_vision(
            attachment_id=attachment_id,
            vision_status="complete",
            vision_summary=result.summary,
            vision_ocr=result.ocr,
            vision_json=result.raw_json,
        )

    def should_ignore_trigger_from_bot(self, message: Message) -> bool:
        user = message.from_user
        if self.settings.respond_to_bots:
            return False
        if not user or not user.is_bot:
            return False
        return str(user.id) not in {str(self.bot_user_id), LEGACY_NEUROMANCER_USER_ID}

    def should_trigger(self, message: Message) -> bool:
        text = str(message.text or message.caption or "")
        lowered = text.casefold()
        if self.is_reply_to_self(message):
            return True
        if self.command_triggered(text):
            return True
        if self.mentions_self(lowered):
            return True
        return False

    def is_reply_to_self(self, message: Message) -> bool:
        reply = message.reply_to_message
        if reply is None or reply.from_user is None:
            return False
        reply_user_id = str(reply.from_user.id)
        return reply_user_id in {str(self.bot_user_id), LEGACY_NEUROMANCER_USER_ID}

    def command_triggered(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped.startswith("/"):
            return False
        first_word = stripped.split(maxsplit=1)[0].casefold()
        command = first_word.split("@", 1)[0]
        return command in {value.casefold() for value in self.settings.command_triggers}

    def mentions_self(self, lowered_text: str) -> bool:
        return any(
            trigger and re.search(rf"(?<!\w)@{re.escape(trigger)}\b", lowered_text)
            for trigger in self.settings.mention_triggers
        )

    async def reply_to_trigger(
        self, message: Message, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        sender_user_id, sender_name, _username, sender_is_bot = user_from_message(message)
        text = message_text(message)
        trigger = TriggerMessage(
            chat_id=str(message.chat_id),
            message_id=str(message.message_id),
            sender_user_id=sender_user_id,
            sender_name=sender_name,
            sent_at=to_iso(message.date),
            text=text,
            clean_text=self.clean_trigger_text(text),
            reply_to_message_id=reply_to_message_id(message),
            was_from_bot=sender_is_bot,
        )
        memory_context = self.memory.load_context(
            chat_id=str(message.chat_id),
            active_user_id=sender_user_id,
            recent_limit=self.settings.recent_message_limit,
        )

        stop_typing = asyncio.Event()
        typing_task = asyncio.create_task(self.keep_typing(context, message.chat_id, stop_typing))
        try:
            reply = await self.agent.generate_reply(trigger, memory_context)
        except HermesError:
            LOGGER.exception("Hermes failed while replying to Telegram message %s", message.message_id)
            reply = AgentReply(reply_text=self.settings.error_reply, media_id=None)
        finally:
            stop_typing.set()
            await wait_quietly(typing_task)

        await self.send_and_record_reply(message, context, reply)

    def clean_trigger_text(self, text: str) -> str:
        cleaned = text.strip()
        if self.command_triggered(cleaned):
            cleaned = cleaned.split(maxsplit=1)[1] if len(cleaned.split(maxsplit=1)) > 1 else ""
        for mention in self.settings.mention_triggers:
            cleaned = re.sub(rf"@{re.escape(mention)}\b", "", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    async def keep_typing(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        chat_id: int | str,
        stop: asyncio.Event,
    ) -> None:
        while not stop.is_set():
            try:
                await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                LOGGER.debug("Failed to send typing action", exc_info=True)
            try:
                await asyncio.wait_for(stop.wait(), timeout=4.5)
            except asyncio.TimeoutError:
                pass

    async def send_and_record_reply(
        self,
        trigger_message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        reply: AgentReply,
    ) -> None:
        asset = self.media_library.get(reply.media_id)
        if asset is not None:
            await self.send_and_record_media_reply(trigger_message, context, reply, asset)
            return

        await self.send_and_record_text_reply(trigger_message, context, reply.reply_text)

    async def send_and_record_text_reply(
        self,
        trigger_message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        reply_text: str,
    ) -> bool:
        chunks = split_text_for_telegram(reply_text, self.settings.max_telegram_chars)
        if not chunks:
            return False

        for index, chunk in enumerate(chunks):
            if index == 0:
                sent = await trigger_message.reply_text(
                    chunk,
                    disable_web_page_preview=True,
                )
                reply_to = str(trigger_message.message_id)
            else:
                sent = await context.bot.send_message(
                    chat_id=trigger_message.chat_id,
                    text=chunk,
                    disable_web_page_preview=True,
                )
                reply_to = None

            if self.bot_user_id is None:
                LOGGER.warning("Bot identity missing; cannot record outgoing message")
                continue
            self.memory.record_outgoing_message(
                sent,
                bot_user_id=self.bot_user_id,
                bot_display_name=self.bot_display_name,
                bot_username=self.bot_username,
                text=chunk,
                reply_to_message_id_value=reply_to,
            )
        return True

    async def send_and_record_media_reply(
        self,
        trigger_message: Message,
        context: ContextTypes.DEFAULT_TYPE,
        reply: AgentReply,
        asset: MediaAsset,
    ) -> None:
        text = reply.reply_text.strip()
        caption = text if text and can_caption(asset) and len(text) <= TELEGRAM_CAPTION_LIMIT else None

        if text and caption is None and asset.send_mode != "media_only":
            await self.send_and_record_text_reply(trigger_message, context, text)
            media_reply_to = None
        else:
            media_reply_to = str(trigger_message.message_id)

        sent = await self.send_media_asset(
            context,
            chat_id=trigger_message.chat_id,
            asset=asset,
            caption=caption,
            reply_to_message_id_value=media_reply_to,
        )
        if self.bot_user_id is None:
            LOGGER.warning("Bot identity missing; cannot record outgoing media")
            return
        self.memory.record_outgoing_media(
            sent,
            bot_user_id=self.bot_user_id,
            bot_display_name=self.bot_display_name,
            bot_username=self.bot_username,
            asset_id=asset.id,
            asset_type=asset.type,
            caption=caption,
            reply_to_message_id_value=media_reply_to,
        )

    async def send_media_asset(
        self,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        chat_id: int | str,
        asset: MediaAsset,
        caption: str | None,
        reply_to_message_id_value: str | None,
    ) -> Message:
        reply_to = int(reply_to_message_id_value) if reply_to_message_id_value else None
        with asset.path.open("rb") as file:
            if asset.type == "photo":
                return await context.bot.send_photo(
                    chat_id=chat_id,
                    photo=file,
                    caption=caption,
                    reply_to_message_id=reply_to,
                )
            if asset.type == "video":
                return await context.bot.send_video(
                    chat_id=chat_id,
                    video=file,
                    caption=caption,
                    reply_to_message_id=reply_to,
                )
            if asset.type == "animation":
                return await context.bot.send_animation(
                    chat_id=chat_id,
                    animation=file,
                    caption=caption,
                    reply_to_message_id=reply_to,
                )
            if asset.type == "document":
                return await context.bot.send_document(
                    chat_id=chat_id,
                    document=file,
                    caption=caption,
                    reply_to_message_id=reply_to,
                )
            if asset.type == "sticker":
                return await context.bot.send_sticker(
                    chat_id=chat_id,
                    sticker=file,
                    reply_to_message_id=reply_to,
                )
        raise RuntimeError(f"Unsupported media asset type: {asset.type}")


def can_caption(asset: MediaAsset) -> bool:
    return asset.type in {"photo", "video", "animation", "document"} and asset.send_mode != "media_only"


def incoming_images_from_message(message: Message) -> list[IncomingImage]:
    photos = getattr(message, "photo", None)
    if photos:
        photo = photos[-1]
        return [
            IncomingImage(
                file_id=photo.file_id,
                file_unique_id=photo.file_unique_id,
                media_type="photo",
                mime_type="image/jpeg",
                extension=".jpg",
                width=photo.width,
                height=photo.height,
                file_size=photo.file_size,
            )
        ]

    document = getattr(message, "document", None)
    if document and document.mime_type and document.mime_type.startswith("image/"):
        extension = mimetypes.guess_extension(document.mime_type) or Path(
            document.file_name or ""
        ).suffix or ".img"
        return [
            IncomingImage(
                file_id=document.file_id,
                file_unique_id=document.file_unique_id,
                media_type="image_document",
                mime_type=document.mime_type,
                extension=extension,
                width=None,
                height=None,
                file_size=document.file_size,
            )
        ]
    return []


def sanitize_path_part(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return sanitized[:120] or "file"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def wait_quietly(task: asyncio.Task[Any]) -> None:
    try:
        await task
    except asyncio.CancelledError:
        raise
    except Exception:
        LOGGER.debug("Background task failed", exc_info=True)


def run_polling(settings: Settings) -> None:
    runtime = NeuromancerTelegramRuntime(settings)
    application = runtime.build_application()
    LOGGER.info("Starting Telegram polling. Python now owns Telegram for this bot token.")
    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=False,
    )
