# Traditional Python Telegram Runtime

This runtime makes Python the only Telegram owner:

1. Telegram update arrives in `neuromancer_bot.telegram_app`.
2. Python checks whether the original Telegram message matches the simple trigger rules.
3. The message is written to `neuromancer_memory.db`.
4. If the message has a Telegram photo/image document, Python temporarily downloads it under `media/incoming/`.
5. OpenRouter vision summarizes the image and stores summary/OCR in `message_attachments`.
6. If the message has voice/audio/video-note media, Python temporarily downloads it, transcribes it with local `faster-whisper`, and updates the same `messages_raw.text` row with `[audio transcript] ...` or `[video-note transcript] ...`.
7. If the original message triggered Neuromancer, the bot builds a compact DB context.
8. The fixed media index is included as optional reaction context.
9. Hermes is invoked with `hermes -z` as the agent/model/tool runtime.
10. Hermes returns `{"reply_text":"...","media_id":null,"delivery":"reply"}`.
11. Python validates any `media_id`, sends text/media, and writes the outgoing result back to SQLite.

`delivery` can be `reply` or `standalone`. Missing/invalid values default to `reply`.

Triggers are intentionally strict:

- reply to Neuromancer
- `/neuromancer`
- `@kazakh_wintermute_bot`

Run from this profile:

```bash
env HERMES_HOME=/Users/karimabilgaziyev/.hermes/profiles/neuromancer \
  /Users/karimabilgaziyev/.hermes/hermes-agent/venv/bin/python -m neuromancer_bot
```

Install `requirements.txt` into the same Python environment that runs the bot.

Expected environment:

- `TELEGRAM_BOT_TOKEN`: Telegram token for Neuromancer.
- `HERMES_HOME`: this profile directory.
- `NEUROMANCER_CHAT_ID`: optional; defaults to `group_profile.chat_id`.
- `NEUROMANCER_RECENT_MESSAGES`: optional; defaults to `40`.
- `NEUROMANCER_HERMES_COMMAND`: optional; defaults to the installed `hermes`.
- `NEUROMANCER_HERMES_TOOLSETS`: optional override for one-shot Hermes calls.
- `NEUROMANCER_MEDIA_INDEX_PATH`: optional; defaults to `media/index.json`.
- `NEUROMANCER_INCOMING_MEDIA_DIR`: optional; defaults to `media/incoming`.
- `NEUROMANCER_VISION_MODEL`: optional; defaults to `config.yaml` auxiliary vision model.
- `NEUROMANCER_VISION_ENABLED`: optional; defaults to true.
- `NEUROMANCER_TRANSCRIPTION_ENABLED`: optional; defaults to true.
- `NEUROMANCER_TRANSCRIPTION_PROVIDER`: optional; defaults to `faster-whisper`.
- `NEUROMANCER_TRANSCRIPTION_MODEL`: optional; defaults to `small`.
- `NEUROMANCER_TRANSCRIPTION_DEVICE`: optional; defaults to `cpu`.
- `NEUROMANCER_TRANSCRIPTION_COMPUTE_TYPE`: optional; defaults to `int8`.
- `NEUROMANCER_TRANSCRIPTION_CPU_THREADS`: optional; defaults to `0`, letting faster-whisper choose.
- `NEUROMANCER_TRANSCRIPTION_BEAM_SIZE`: optional; defaults to `5`.
- `NEUROMANCER_TRANSCRIPTION_LANGUAGE`: optional language hint like `ru`; unset means auto-detect.
- `NEUROMANCER_TRANSCRIPTION_MAX_SECONDS`: optional; defaults to `300`.
- `NEUROMANCER_TRANSCRIPTION_MAX_BYTES`: optional; defaults to `25000000`.
- `NEUROMANCER_FFMPEG_COMMAND`: optional; defaults to `ffmpeg`.
- `NEUROMANCER_KEEP_INCOMING_MEDIA`: optional; defaults to false, deleting downloaded incoming media after vision/transcription processing.

Voice/audio/video-note messages are transcribed for memory, but trigger detection stays simple for now: a spoken mention inside audio is not enough to wake Neuromancer. The original Telegram message still has to be a reply to Neuromancer, `/neuromancer`, or `@kazakh_wintermute_bot`.

The runtime never updates curated `group_profile.profile`, `group_profile.raw_profile`,
`members.profile`, `members.raw_profile`, or manually chosen `primary_name` values.
