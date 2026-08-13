# Neuromancer Hermes Profile

Neuromancer is a private Telegram group-chat AI agent. The current architecture is a traditional Python Telegram bot that owns Telegram directly, stores memory in SQLite, and uses Hermes as the language-model/tool runtime.

The important split:

- Python owns Telegram updates, replies, and database writes.
- SQLite owns durable chat memory and manually curated group/member context.
- Hermes owns model inference, `SOUL.md`, and tool use.

## Runtime Flow

```text
Telegram group message
  -> neuromancer_bot.telegram_app
  -> decide whether the original Telegram message should trigger Neuromancer
  -> write incoming message to neuromancer_memory.db
  -> download/analyze Telegram images with OpenRouter vision, when present
  -> download/transcribe Telegram voice/audio/video-note messages, when present
  -> if triggered, build compact context from SQLite
  -> call Hermes with hermes -z
  -> send Telegram reply
  -> write outgoing Neuromancer message to SQLite
```

Python is the only intended Telegram consumer. The Hermes Telegram gateway should stay stopped for this profile, otherwise two processes can fight over the same bot token.

## Main Files

- `SOUL.md`: Neuromancer's personality, behavioral rules, tone, and identity.
- `neuromancer_memory.db`: SQLite memory database.
- `neuromancer_bot/telegram_app.py`: Telegram polling app, trigger detection, typing indicator, reply sending.
- `neuromancer_bot/memory.py`: SQLite memory API for messages, aliases, and member activity stats.
- `neuromancer_bot/agent.py`: Hermes one-shot subprocess caller and prompt/context builder.
- `neuromancer_bot/media_library.py`: fixed media manifest loader and validator.
- `neuromancer_bot/vision.py`: OpenRouter vision preprocessing for incoming Telegram images.
- `neuromancer_bot/transcription.py`: local `faster-whisper` transcription and `ffmpeg` audio extraction helpers.
- `neuromancer_bot/config.py`: Runtime settings loaded from environment and `.env`.
- `bin/neuromancer-bot`: Convenience launcher for the Python bot.
- `media/index.json`: curated fixed media reactions Hermes may choose by id.
- `scripts/import_telegram_export.py`: One-time/import utility for Telegram Desktop JSON exports.
- `scripts/search_memory.py`: Read-only CLI for searching stored Telegram messages.
- `scripts/extract_web_context.py`: Local readable-text extractor for URLs found by web search.
- `docs/traditional-python-bot.md`: Short operational notes for this runtime.
- `plugins/neuromancer-memory/`: Earlier Hermes plugin experiment. It is kept as reference but disabled.
- `plugins/neuromancer-db-search/`: Read-only Hermes tools for searching `messages_raw` and `messages_fts`.
- `plugins/neuromancer-web-extract/`: Hermes web extraction provider/tool for DuckDuckGo search results.

## Memory Database

`neuromancer_memory.db` contains the long-term local context for the group.

Core tables:

- `group_profile`: one manually curated row describing the private group.
- `members`: manually curated member profiles plus live activity stats.
- `member_aliases`: aliases for each known member.
- `messages_raw`: raw Telegram message log, including timestamps, sender snapshots, message type, reply target, trigger flag, and raw Telegram payload.
- `message_attachments`: incoming image metadata, local paths, hashes, vision summaries, OCR, and raw vision JSON.
- `message_transcriptions`: incoming voice/audio/video-note metadata, transcription status, transcript text, and raw transcription JSON.
- `media_sends`: outgoing fixed-media send log, keyed by Telegram message id.
- `messages_fts`: full-text index over raw message text and sender snapshots.

Manual ownership boundary:

- The bot must not rewrite `group_profile.profile` or `group_profile.raw_profile`.
- The bot must not rewrite `members.profile` or `members.raw_profile`.
- Existing curated `members.primary_name` values are left alone.

Live-owned fields:

- new `messages_raw` rows
- new `message_attachments` rows for downloaded Telegram images
- new `message_transcriptions` rows for voice/audio/video-note messages
- enriched `messages_raw.text` values for transcribed voice/audio/video-note messages
- `members.username` when it was previously missing
- `members.is_bot` when Telegram confirms a bot
- `members.first_seen_at`
- `members.last_seen_at`
- `members.message_count`
- `member_aliases` with source `live`

## Database Search Tools

Hermes can search old group messages through the enabled `neuromancer-db-search` plugin. These tools are read-only and live in the existing `memory` toolset:

- `neuromancer_memory_search`: search `messages_fts` or exact substrings in `messages_raw`, with optional sender/date filters.
- `neuromancer_message_context`: fetch one Telegram/raw message id plus nearby messages.
- `neuromancer_user_messages`: fetch recent messages from a specific Telegram user id.

Manual test:

```bash
python3 scripts/search_memory.py search "exact words here" --mode exact --limit 5
python3 scripts/search_memory.py context tg:8122 --window 5
```

## Trigger Behavior

Every message in the target group is stored. Neuromancer only replies when a message appears to be addressed to it.

Default triggers:

- reply to Neuromancer
- `/neuromancer`
- `@kazakh_wintermute_bot`

By default, messages from other bots are stored but do not trigger a reply. This prevents loops with Ikram's bot while still preserving the chat transcript.

## Hermes Role

Hermes is used as the agent brain through one-shot CLI calls:

```bash
hermes --accept-hooks -z "<constructed prompt>"
```

The constructed prompt includes:

- instruction to follow `SOUL.md`
- database mutability boundary
- group profile
- known member profiles and aliases
- active speaker profile
- recent messages, oldest to newest
- vision summaries/OCR for recent image attachments
- audio/video-note transcripts already stored as `[audio transcript] ...` or `[video-note transcript] ...`
- the triggering message
- compact fixed media options from `media/index.json`

Hermes returns a JSON envelope:

```json
{"reply_text":"telegram reply text","media_id":null,"delivery":"reply"}
```

If a curated reaction image fits, `media_id` may be one approved id from `media/index.json`. Python validates the id before sending anything. Hermes does not get to send arbitrary file paths.

`delivery` controls whether the outgoing text/media is attached to the triggering message or sent as an ordinary group message:

- `reply`: default; attach the first text chunk or captioned/media-only send to the triggering message.
- `standalone`: send text/media without `reply_to_message_id`.

Because Hermes runs inside this profile, it still has the configured OpenRouter model and enabled tools. Telegram transport is no longer handled by Hermes.

## Web Search

Hermes web search is configured to use DuckDuckGo through the `ddgs` backend:

```yaml
web:
  backend: ddgs
  search_backend: ddgs
  extract_backend: neuromancer-readable
```

DuckDuckGo search returns titles, URLs, and snippets. The local `neuromancer-web-extract` plugin turns selected result URLs into readable page text so Neuromancer can answer from the actual source instead of only the search snippet.

The plugin exposes two paths:

- normal Hermes `web_extract`, routed through the `neuromancer-readable` provider
- direct `web_extract_readable`, registered in the existing `web` toolset as a clear fallback after `web_search`

The extractor handles public HTML, text, markdown, XML, and JSON pages. It blocks non-HTTP(S), embedded credentials, credential-like query parameters, private/internal hosts, and URLs blocked by Hermes website policy. It does not handle PDFs or JavaScript-only pages.

Manual test:

```bash
python3 scripts/extract_web_context.py https://example.com
```

## Incoming Vision

When a Telegram message contains a photo or image document, Python temporarily downloads the image under:

```text
media/incoming/YYYY/MM/
```

That directory is ignored by git because it can contain private chat artifacts.

For each downloaded image, Python calls OpenRouter directly as a sensory preprocessing step. The default vision model is read from `config.yaml`:

```yaml
auxiliary:
  vision:
    provider: openrouter
    model: google/gemini-2.5-flash
```

The vision result is stored in `message_attachments`:

- Telegram file id and unique id
- local path
- mime type, dimensions, file size, sha256
- `vision_status`
- `vision_summary`
- `vision_ocr`
- raw structured vision JSON

If an image message triggers Neuromancer, the vision step runs before the Hermes reply, so Hermes sees the summary and OCR in recent context. Non-triggering images are still stored and analyzed for future memory.

By default, local incoming media files are deleted after the vision attempt to keep VPS disk usage low. The database keeps Telegram file ids, hashes, dimensions, status, summaries/OCR, and errors. Set this only if you want a local archive for re-running vision later:

```env
NEUROMANCER_KEEP_INCOMING_MEDIA=true
```

## Incoming Audio

When a Telegram message contains a voice message, audio file, or video note, Python records the original Telegram message first, then tries to transcribe it.

By default this uses local `faster-whisper` with model `small`, `cpu`, and `int8` compute. The first transcription on a fresh machine may take longer because the model has to be downloaded and cached.

Stored message text is enriched in `messages_raw.text`:

```text
[audio transcript] transcribed voice or audio text
[video-note transcript] transcribed round-video audio text
```

Audio-file captions are preserved under a `[caption]` line after the transcript. Because the same `messages_raw` row is updated, transcripts automatically appear in recent context and in `messages_fts` search.

Video notes are downloaded as Telegram video files, then converted to `.mp3` through `ffmpeg` before transcription. Voice/audio files are sent directly to local `faster-whisper` when their extension is already supported; unknown formats are normalized through `ffmpeg`.

Trigger behavior is intentionally simple: Neuromancer decides whether to reply before transcription. A voice or video-note message will trigger only if the original Telegram message was a reply to Neuromancer. A spoken mention inside audio is remembered after transcription, but does not wake the bot yet.

By default, downloaded audio/video-note files and converted `.mp3` files are deleted after processing. The database keeps Telegram file ids, hashes, durations, transcript text, statuses, raw transcription JSON, and any error.

Required system package for video notes and unknown audio formats:

```bash
ffmpeg
```

## Fixed Media

Fixed reactions live under `media/` and are indexed by `media/index.json`.

Each asset has:

- `id`: stable id Hermes may return.
- `path`: local file path.
- `type`: Telegram media type, currently supporting `photo`, `video`, `animation`, `document`, and `sticker`.
- `description`: what the media means.
- `use_when`: situations where it fits.
- `avoid_when`: situations where it would be wrong or overused.
- `send_mode`: `with_optional_text`, `caption`, or `media_only`.

Outgoing media is recorded twice:

- as a `messages_raw` row like `[photo:asset_id] caption`
- as a `media_sends` row with `asset_id`, Telegram message id, caption, and raw payload

## Running

From this directory:

```bash
./bin/neuromancer-bot
```

Install Python dependencies into the same Python environment that runs the bot:

```bash
python3 -m pip install -r requirements.txt
```

If your runtime Python is managed by `uv` or a service-specific venv, install `requirements.txt` into that exact interpreter.

Equivalent explicit command:

```bash
env HERMES_HOME=/Users/karimabilgaziyev/.hermes/profiles/neuromancer \
  /Users/karimabilgaziyev/.hermes/hermes-agent/venv/bin/python -m neuromancer_bot
```

Required environment:

- `TELEGRAM_BOT_TOKEN`: Telegram bot token.

Useful optional environment:

- `NEUROMANCER_CHAT_ID`: target group chat id. Defaults to `group_profile.chat_id`.
- `NEUROMANCER_RECENT_MESSAGES`: number of recent messages sent to Hermes. Defaults to `40`.
- `NEUROMANCER_HERMES_COMMAND`: Hermes executable path. Defaults to installed `hermes`.
- `NEUROMANCER_HERMES_TOOLSETS`: optional toolset override for Hermes one-shot calls.
- `NEUROMANCER_HERMES_MODEL`: optional model override.
- `NEUROMANCER_HERMES_PROVIDER`: optional provider override.
- `NEUROMANCER_HERMES_REASONING`: optional reasoning override.
- `NEUROMANCER_MEDIA_INDEX_PATH`: optional path to the fixed media manifest.
- `NEUROMANCER_INCOMING_MEDIA_DIR`: optional path for downloaded incoming media.
- `NEUROMANCER_VISION_ENABLED`: defaults to true.
- `NEUROMANCER_VISION_PROVIDER`: defaults to `config.yaml` auxiliary vision provider.
- `NEUROMANCER_VISION_MODEL`: defaults to `config.yaml` auxiliary vision model.
- `NEUROMANCER_VISION_BASE_URL`: defaults to OpenRouter chat completions.
- `NEUROMANCER_VISION_TIMEOUT`: defaults to `60`.
- `NEUROMANCER_TRANSCRIPTION_ENABLED`: defaults to true.
- `NEUROMANCER_TRANSCRIPTION_PROVIDER`: defaults to `faster-whisper`.
- `NEUROMANCER_TRANSCRIPTION_MODEL`: defaults to `small`.
- `NEUROMANCER_TRANSCRIPTION_DEVICE`: defaults to `cpu`.
- `NEUROMANCER_TRANSCRIPTION_COMPUTE_TYPE`: defaults to `int8`.
- `NEUROMANCER_TRANSCRIPTION_CPU_THREADS`: defaults to `0`, letting faster-whisper choose.
- `NEUROMANCER_TRANSCRIPTION_BEAM_SIZE`: defaults to `5`.
- `NEUROMANCER_TRANSCRIPTION_LANGUAGE`: optional language hint like `ru`; unset means auto-detect.
- `NEUROMANCER_TRANSCRIPTION_BASE_URL`: only used when `NEUROMANCER_TRANSCRIPTION_PROVIDER=openai`.
- `NEUROMANCER_TRANSCRIPTION_TIMEOUT`: defaults to `90`.
- `NEUROMANCER_TRANSCRIPTION_MAX_SECONDS`: defaults to `300`.
- `NEUROMANCER_TRANSCRIPTION_MAX_BYTES`: defaults to `25000000`.
- `NEUROMANCER_FFMPEG_COMMAND`: defaults to `ffmpeg`.
- `NEUROMANCER_FFMPEG_TIMEOUT`: defaults to `60`.
- `NEUROMANCER_KEEP_INCOMING_MEDIA`: defaults to false; deletes downloaded incoming media after vision/transcription processing.
- `NEUROMANCER_RESPOND_TO_BOTS`: defaults to false.
- `NEUROMANCER_LOG_LEVEL`: defaults to `INFO`.

The bot loads `.env` from this profile by default.

## Operational Notes

Check that the Hermes gateway is not running:

```bash
env HERMES_HOME=/Users/karimabilgaziyev/.hermes/profiles/neuromancer hermes gateway list
```

Stop it if needed:

```bash
env HERMES_HOME=/Users/karimabilgaziyev/.hermes/profiles/neuromancer hermes gateway stop
```

Check database health:

```bash
sqlite3 neuromancer_memory.db 'PRAGMA integrity_check;'
```

Import a Telegram Desktop export:

```bash
python3 scripts/import_telegram_export.py result.json
```

## Why This Architecture

The earlier Hermes plugin approach worked, but it had to hook into Hermes' Telegram adapter. This version makes memory a first-class part of the application:

- all Telegram updates pass through code owned by this repo
- memory writes happen before any model decision
- database write boundaries are explicit and testable
- Telegram token ownership is simple
- Hermes remains useful as the model and tool runtime
