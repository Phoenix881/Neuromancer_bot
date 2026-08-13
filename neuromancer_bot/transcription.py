"""Audio transcription for Telegram voice/audio messages and video notes."""

from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_DIRECT_EXTENSIONS = {
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".ogg",
    ".wav",
    ".webm",
}


class TranscriptionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    raw_json: str


class FasterWhisperTranscriber:
    def __init__(
        self,
        *,
        model: str,
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 0,
        beam_size: int = 5,
        language: str | None = None,
    ):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.beam_size = beam_size
        self.language = language
        self._model: Any | None = None
        self._lock = threading.Lock()

    def transcribe(self, path: Path) -> TranscriptionResult:
        if not path.exists():
            raise TranscriptionError(f"Audio file does not exist: {path}")

        model = self._load_model()
        kwargs: dict[str, Any] = {"beam_size": self.beam_size}
        if self.language:
            kwargs["language"] = self.language

        try:
            segments_iter, info = model.transcribe(str(path), **kwargs)
            segments = list(segments_iter)
        except Exception as exc:
            raise TranscriptionError(f"faster-whisper transcription failed: {exc}") from exc

        transcript = " ".join(str(segment.text).strip() for segment in segments).strip()
        if not transcript:
            raise TranscriptionError("faster-whisper returned empty transcript")

        language = getattr(info, "language", None)
        payload = {
            "provider": "faster-whisper",
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "language": language,
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
            "segments": [
                {
                    "id": getattr(segment, "id", index),
                    "start": getattr(segment, "start", None),
                    "end": getattr(segment, "end", None),
                    "text": str(segment.text).strip(),
                }
                for index, segment in enumerate(segments)
            ],
        }
        return TranscriptionResult(
            text=transcript,
            language=str(language) if language else None,
            raw_json=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise TranscriptionError(
                    "faster-whisper is not installed; run pip install -r requirements.txt"
                ) from exc

            kwargs: dict[str, Any] = {
                "device": self.device,
                "compute_type": self.compute_type,
            }
            if self.cpu_threads > 0:
                kwargs["cpu_threads"] = self.cpu_threads
            try:
                self._model = WhisperModel(self.model_name, **kwargs)
            except Exception as exc:
                raise TranscriptionError(f"failed to load faster-whisper model: {exc}") from exc
            return self._model


class OpenAIWhisperTranscriber:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout_seconds: float,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def transcribe(self, path: Path) -> TranscriptionResult:
        api_key = openai_api_key()
        if not api_key:
            raise TranscriptionError("OPENAI_API_KEY is not set")
        if not path.exists():
            raise TranscriptionError(f"Audio file does not exist: {path}")

        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body, boundary = multipart_form_data(
            fields={
                "model": self.model,
                "response_format": "verbose_json",
            },
            file_field="file",
            file_path=path,
            file_content_type=content_type,
        )
        request = urllib.request.Request(
            self.base_url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise TranscriptionError(f"Whisper HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise TranscriptionError(f"Whisper request failed: {exc}") from exc

        return parse_transcription_response(raw_response)


def openai_api_key() -> str | None:
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_AUDIO_API_KEY",
        "NEUROMANCER_OPENAI_API_KEY",
    ):
        value = os.environ.get(name)
        if value:
            return value
    return None


def multipart_form_data(
    *,
    fields: dict[str, str],
    file_field: str,
    file_path: Path,
    file_content_type: str,
) -> tuple[bytes, str]:
    boundary = f"----NeuromancerBoundary{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.extend(
            [
                f"--{boundary}\r\n".encode("ascii"),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                str(value).encode("utf-8"),
                b"\r\n",
            ]
        )

    filename = file_path.name
    chunks.extend(
        [
            f"--{boundary}\r\n".encode("ascii"),
            (
                f'Content-Disposition: form-data; name="{file_field}"; '
                f'filename="{filename}"\r\n'
            ).encode("utf-8"),
            f"Content-Type: {file_content_type}\r\n\r\n".encode("ascii"),
            file_path.read_bytes(),
            b"\r\n",
            f"--{boundary}--\r\n".encode("ascii"),
        ]
    )
    return b"".join(chunks), boundary


def parse_transcription_response(raw_response: str) -> TranscriptionResult:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise TranscriptionError("Whisper response was not valid JSON") from exc
    if not isinstance(data, dict):
        raise TranscriptionError("Whisper response was not a JSON object")

    text = str(data.get("text") or "").strip()
    if not text:
        raise TranscriptionError("Whisper response did not contain transcript text")

    language_value = data.get("language")
    language = str(language_value).strip() if language_value else None
    raw_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return TranscriptionResult(text=text, language=language, raw_json=raw_json)


def needs_mp3_conversion(path: Path, *, media_type: str) -> bool:
    if media_type == "video_note":
        return True
    return path.suffix.casefold() not in SUPPORTED_DIRECT_EXTENSIONS


def convert_to_mp3(
    *,
    source_path: Path,
    target_path: Path,
    ffmpeg_command: str,
    timeout_seconds: float,
) -> Path:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_command,
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "64k",
        str(target_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        raise TranscriptionError(f"ffmpeg command not found: {ffmpeg_command}") from exc
    except subprocess.TimeoutExpired as exc:
        raise TranscriptionError(f"ffmpeg timed out after {timeout_seconds:.0f}s") from exc

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise TranscriptionError(f"ffmpeg failed with exit code {completed.returncode}: {detail}")
    if not target_path.exists():
        raise TranscriptionError(f"ffmpeg did not create output file: {target_path}")
    return target_path


def transcription_display_text(
    *,
    media_type: str,
    transcript: str,
    caption: str | None = None,
) -> str:
    prefix = "[video-note transcript]" if media_type == "video_note" else "[audio transcript]"
    text = transcript.strip()
    result = f"{prefix} {text}" if text else prefix
    caption_text = (caption or "").strip()
    if caption_text:
        result = f"{result}\n[caption] {caption_text}"
    return result


def transcription_skip_text(
    *,
    media_type: str,
    reason: str,
    caption: str | None = None,
) -> str:
    prefix = (
        "[video-note transcript unavailable]"
        if media_type == "video_note"
        else "[audio transcript unavailable]"
    )
    result = f"{prefix} {reason}"
    caption_text = (caption or "").strip()
    if caption_text:
        result = f"{result}\n[caption] {caption_text}"
    return result
