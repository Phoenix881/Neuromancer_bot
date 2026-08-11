"""Entrypoint for `python -m neuromancer_bot`."""

from __future__ import annotations

import logging

from .config import load_settings
from .telegram_app import run_polling


def main() -> None:
    settings = load_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_polling(settings)


if __name__ == "__main__":
    main()
