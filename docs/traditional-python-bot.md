# Traditional Python Telegram Runtime

This runtime makes Python the only Telegram owner:

1. Telegram update arrives in `neuromancer_bot.telegram_app`.
2. The message is written to `neuromancer_memory.db`.
3. If the message has a Telegram photo/image document, Python downloads it under `media/incoming/`.
4. OpenRouter vision summarizes the image and stores summary/OCR in `message_attachments`.
5. If it triggers Neuromancer, the bot builds a compact DB context.
6. The fixed media index is included as optional reaction context.
7. Hermes is invoked with `hermes -z` as the agent/model/tool runtime.
8. Hermes returns `{"reply_text":"...","media_id":null}`.
9. Python validates any `media_id`, sends text/media, and writes the outgoing result back to SQLite.

Triggers are intentionally strict:

- reply to Neuromancer
- `/neuromancer`
- `@kazakh_wintermute_bot`

Run from this profile:

```bash
env HERMES_HOME=/Users/karimabilgaziyev/.hermes/profiles/neuromancer \
  /Users/karimabilgaziyev/.hermes/hermes-agent/venv/bin/python -m neuromancer_bot
```

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

The runtime never updates curated `group_profile.profile`, `group_profile.raw_profile`,
`members.profile`, `members.raw_profile`, or manually chosen `primary_name` values.
