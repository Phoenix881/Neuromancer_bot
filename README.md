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
  -> write incoming message to neuromancer_memory.db
  -> download/analyze Telegram images with OpenRouter vision, when present
  -> decide whether Neuromancer should reply
  -> build compact context from SQLite
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
- `neuromancer_bot/config.py`: Runtime settings loaded from environment and `.env`.
- `bin/neuromancer-bot`: Convenience launcher for the Python bot.
- `media/index.json`: curated fixed media reactions Hermes may choose by id.
- `scripts/import_telegram_export.py`: One-time/import utility for Telegram Desktop JSON exports.
- `docs/traditional-python-bot.md`: Short operational notes for this runtime.
- `plugins/neuromancer-memory/`: Earlier Hermes plugin experiment. It is kept as reference but disabled.

## Memory Database

`neuromancer_memory.db` contains the long-term local context for the group.

Core tables:

- `group_profile`: one manually curated row describing the private group.
- `members`: manually curated member profiles plus live activity stats.
- `member_aliases`: aliases for each known member.
- `messages_raw`: raw Telegram message log, including timestamps, sender snapshots, message type, reply target, trigger flag, and raw Telegram payload.
- `message_attachments`: incoming image metadata, local paths, hashes, vision summaries, OCR, and raw vision JSON.
- `media_sends`: outgoing fixed-media send log, keyed by Telegram message id.
- `messages_fts`: full-text index over raw message text and sender snapshots.

Manual ownership boundary:

- The bot must not rewrite `group_profile.profile` or `group_profile.raw_profile`.
- The bot must not rewrite `members.profile` or `members.raw_profile`.
- Existing curated `members.primary_name` values are left alone.

Live-owned fields:

- new `messages_raw` rows
- new `message_attachments` rows for downloaded Telegram images
- `members.username` when it was previously missing
- `members.is_bot` when Telegram confirms a bot
- `members.first_seen_at`
- `members.last_seen_at`
- `members.message_count`
- `member_aliases` with source `live`

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
- the triggering message
- compact fixed media options from `media/index.json`

Hermes returns a JSON envelope:

```json
{"reply_text":"telegram reply text","media_id":null}
```

If a curated reaction image fits, `media_id` may be one approved id from `media/index.json`. Python validates the id before sending anything. Hermes does not get to send arbitrary file paths.

Because Hermes runs inside this profile, it still has the configured OpenRouter model and enabled tools. Telegram transport is no longer handled by Hermes.

## Incoming Vision

When a Telegram message contains a photo or image document, Python downloads the image under:

```text
media/incoming/YYYY/MM/
```

That directory is ignored by git because it contains private chat artifacts.

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
- `NEUROMANCER_INCOMING_MEDIA_DIR`: optional path for downloaded incoming images.
- `NEUROMANCER_VISION_ENABLED`: defaults to true.
- `NEUROMANCER_VISION_PROVIDER`: defaults to `config.yaml` auxiliary vision provider.
- `NEUROMANCER_VISION_MODEL`: defaults to `config.yaml` auxiliary vision model.
- `NEUROMANCER_VISION_BASE_URL`: defaults to OpenRouter chat completions.
- `NEUROMANCER_VISION_TIMEOUT`: defaults to `60`.
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
