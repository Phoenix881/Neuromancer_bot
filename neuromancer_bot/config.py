"""Runtime settings for the standalone Neuromancer Telegram bot."""

from __future__ import annotations

import os
import shlex
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CHAT_ID = "-1003279704534"
DEFAULT_DB_FILENAME = "neuromancer_memory.db"
DEFAULT_VISION_PROVIDER = "openrouter"
DEFAULT_VISION_MODEL = "google/gemini-2.5-flash"
DEFAULT_VISION_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TRANSCRIPTION_PROVIDER = "faster-whisper"
DEFAULT_TRANSCRIPTION_MODEL = "small"
DEFAULT_TRANSCRIPTION_BASE_URL = "https://api.openai.com/v1/audio/transcriptions"
DEFAULT_TRANSCRIPTION_DEVICE = "cpu"
DEFAULT_TRANSCRIPTION_COMPUTE_TYPE = "int8"
DEFAULT_COMMAND_TRIGGERS = ("/neuromancer",)
DEFAULT_MENTION_TRIGGERS = ("kazakh_wintermute_bot",)
TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


@dataclass(frozen=True)
class Settings:
    hermes_home: Path
    db_path: Path
    media_index_path: Path
    incoming_media_dir: Path
    telegram_bot_token: str
    target_chat_id: str
    hermes_command: str
    hermes_timeout_seconds: float
    recent_message_limit: int
    respond_to_bots: bool
    command_triggers: tuple[str, ...]
    mention_triggers: tuple[str, ...]
    hermes_model: str | None
    hermes_provider: str | None
    hermes_toolsets: str | None
    hermes_reasoning: str | None
    max_telegram_chars: int
    vision_enabled: bool
    vision_provider: str
    vision_model: str
    vision_base_url: str
    vision_timeout_seconds: float
    transcription_enabled: bool
    transcription_provider: str
    transcription_model: str
    transcription_base_url: str
    transcription_timeout_seconds: float
    transcription_max_seconds: int
    transcription_max_bytes: int
    transcription_device: str
    transcription_compute_type: str
    transcription_cpu_threads: int
    transcription_beam_size: int
    transcription_language: str | None
    ffmpeg_command: str
    ffmpeg_timeout_seconds: float
    keep_incoming_media: bool
    error_reply: str
    log_level: str


def _profile_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if value:
            try:
                parts = shlex.split(value, posix=True)
            except ValueError:
                parts = []
            if len(parts) == 1:
                value = parts[0]
            elif (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
        os.environ[key] = value


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    return default


def _env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _csv_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or default


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _default_hermes_command() -> str:
    found = shutil.which("hermes")
    if found:
        return found
    bundled = Path.home() / ".hermes" / "hermes-agent" / "hermes"
    if bundled.exists():
        return str(bundled)
    return "hermes"


def _chat_id_from_db(db_path: Path) -> str | None:
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT chat_id FROM group_profile WHERE id = 1").fetchone()
    except sqlite3.Error:
        return None
    return str(row[0]) if row and row[0] is not None else None


def _config_vision_defaults(config_path: Path) -> dict[str, str]:
    defaults: dict[str, str] = {}
    if not config_path.exists():
        return defaults

    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return defaults

    in_auxiliary = False
    in_vision = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent == 0:
            in_auxiliary = stripped == "auxiliary:"
            in_vision = False
            continue
        if in_auxiliary and indent == 2:
            in_vision = stripped == "vision:"
            continue
        if in_auxiliary and in_vision and indent == 4 and ":" in stripped:
            key, value = stripped.split(":", 1)
            defaults[key.strip()] = value.strip().strip("'\"")
    return defaults


def load_settings() -> Settings:
    hermes_home = Path(os.environ.get("HERMES_HOME") or _profile_root()).resolve()
    env_path = Path(os.environ.get("NEUROMANCER_ENV_FILE") or hermes_home / ".env")
    if _env_bool("NEUROMANCER_LOAD_ENV", True):
        _load_env_file(env_path)

    db_path = Path(os.environ.get("NEUROMANCER_DB_PATH") or hermes_home / DEFAULT_DB_FILENAME)
    if not db_path.is_absolute():
        db_path = hermes_home / db_path

    media_index_path = Path(
        os.environ.get("NEUROMANCER_MEDIA_INDEX_PATH") or hermes_home / "media" / "index.json"
    )
    if not media_index_path.is_absolute():
        media_index_path = hermes_home / media_index_path

    incoming_media_dir = Path(
        os.environ.get("NEUROMANCER_INCOMING_MEDIA_DIR")
        or hermes_home / "media" / "incoming"
    )
    if not incoming_media_dir.is_absolute():
        incoming_media_dir = hermes_home / incoming_media_dir

    vision_defaults = _config_vision_defaults(hermes_home / "config.yaml")

    telegram_bot_token = _first_env(
        "TELEGRAM_BOT_TOKEN",
        "HERMES_TELEGRAM_BOT_TOKEN",
        "TELEGRAM_TOKEN",
        "BOT_TOKEN",
    )
    if not telegram_bot_token:
        raise RuntimeError(
            "Missing Telegram bot token. Set TELEGRAM_BOT_TOKEN in .env or the environment."
        )

    target_chat_id = (
        _first_env("NEUROMANCER_CHAT_ID", "TELEGRAM_CHAT_ID", "TELEGRAM_ALLOWED_CHAT_ID")
        or _chat_id_from_db(db_path)
        or DEFAULT_CHAT_ID
    )

    return Settings(
        hermes_home=hermes_home,
        db_path=db_path,
        media_index_path=media_index_path,
        incoming_media_dir=incoming_media_dir,
        telegram_bot_token=telegram_bot_token,
        target_chat_id=str(target_chat_id),
        hermes_command=os.environ.get("NEUROMANCER_HERMES_COMMAND") or _default_hermes_command(),
        hermes_timeout_seconds=_env_float("NEUROMANCER_HERMES_TIMEOUT", 180.0, minimum=10.0),
        recent_message_limit=_env_int("NEUROMANCER_RECENT_MESSAGES", 40, minimum=5),
        respond_to_bots=_env_bool("NEUROMANCER_RESPOND_TO_BOTS", False),
        command_triggers=tuple(trigger.casefold() for trigger in DEFAULT_COMMAND_TRIGGERS),
        mention_triggers=tuple(trigger.casefold() for trigger in DEFAULT_MENTION_TRIGGERS),
        hermes_model=os.environ.get("NEUROMANCER_HERMES_MODEL"),
        hermes_provider=os.environ.get("NEUROMANCER_HERMES_PROVIDER"),
        hermes_toolsets=os.environ.get("NEUROMANCER_HERMES_TOOLSETS"),
        hermes_reasoning=os.environ.get("NEUROMANCER_HERMES_REASONING"),
        max_telegram_chars=_env_int("NEUROMANCER_MAX_TELEGRAM_CHARS", 3900, minimum=1000),
        vision_enabled=_env_bool("NEUROMANCER_VISION_ENABLED", True),
        vision_provider=(
            os.environ.get("NEUROMANCER_VISION_PROVIDER")
            or vision_defaults.get("provider")
            or DEFAULT_VISION_PROVIDER
        ),
        vision_model=(
            os.environ.get("NEUROMANCER_VISION_MODEL")
            or vision_defaults.get("model")
            or DEFAULT_VISION_MODEL
        ),
        vision_base_url=os.environ.get("NEUROMANCER_VISION_BASE_URL") or DEFAULT_VISION_BASE_URL,
        vision_timeout_seconds=_env_float("NEUROMANCER_VISION_TIMEOUT", 60.0, minimum=5.0),
        transcription_enabled=_env_bool("NEUROMANCER_TRANSCRIPTION_ENABLED", True),
        transcription_provider=(
            os.environ.get("NEUROMANCER_TRANSCRIPTION_PROVIDER")
            or DEFAULT_TRANSCRIPTION_PROVIDER
        ).casefold(),
        transcription_model=(
            os.environ.get("NEUROMANCER_TRANSCRIPTION_MODEL")
            or DEFAULT_TRANSCRIPTION_MODEL
        ),
        transcription_base_url=(
            os.environ.get("NEUROMANCER_TRANSCRIPTION_BASE_URL")
            or DEFAULT_TRANSCRIPTION_BASE_URL
        ),
        transcription_timeout_seconds=_env_float(
            "NEUROMANCER_TRANSCRIPTION_TIMEOUT",
            90.0,
            minimum=5.0,
        ),
        transcription_max_seconds=_env_int(
            "NEUROMANCER_TRANSCRIPTION_MAX_SECONDS",
            300,
            minimum=1,
        ),
        transcription_max_bytes=_env_int(
            "NEUROMANCER_TRANSCRIPTION_MAX_BYTES",
            25_000_000,
            minimum=1_000,
        ),
        transcription_device=(
            os.environ.get("NEUROMANCER_TRANSCRIPTION_DEVICE")
            or DEFAULT_TRANSCRIPTION_DEVICE
        ),
        transcription_compute_type=(
            os.environ.get("NEUROMANCER_TRANSCRIPTION_COMPUTE_TYPE")
            or DEFAULT_TRANSCRIPTION_COMPUTE_TYPE
        ),
        transcription_cpu_threads=_env_int(
            "NEUROMANCER_TRANSCRIPTION_CPU_THREADS",
            0,
            minimum=0,
        ),
        transcription_beam_size=_env_int(
            "NEUROMANCER_TRANSCRIPTION_BEAM_SIZE",
            5,
            minimum=1,
        ),
        transcription_language=os.environ.get("NEUROMANCER_TRANSCRIPTION_LANGUAGE") or None,
        ffmpeg_command=os.environ.get("NEUROMANCER_FFMPEG_COMMAND") or "ffmpeg",
        ffmpeg_timeout_seconds=_env_float("NEUROMANCER_FFMPEG_TIMEOUT", 60.0, minimum=5.0),
        keep_incoming_media=_env_bool("NEUROMANCER_KEEP_INCOMING_MEDIA", False),
        error_reply=os.environ.get("NEUROMANCER_ERROR_REPLY")
        or "Hermes choked for a second. I logged it instead of pretending that was wisdom.",
        log_level=os.environ.get("NEUROMANCER_LOG_LEVEL", "INFO").upper(),
    )
