"""OpenRouter vision preprocessing for Telegram images."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
VISION_PROMPT = """Ты — вспомогательная модель зрения для AI-агента Neuromancer.

Твоя единственная задача — анализировать изображения и передавать Neuromancer точную и полезную информацию о том, что на них изображено. Ты не общаешься с пользователем напрямую. Neuromancer использует твой анализ как зрение.

Будь готов к любым изображениям, которые могут появиться в дружеском Telegram-чате: обычным фотографиям, мемам, shitposting, абсурдным и сюрреалистичным изображениям, странному юмору, скриншотам переписок, социальных сетей, игр, программ, терминала, кода, ошибок, графиков и другим техническим материалам.

### Главное правило

Не ограничивайся буквальным описанием изображения. Старайся понять его смысл и контекст.

Если изображение является мемом, шуткой или абсурдным shitpost'ом, объясни Neuromancer:

* что буквально изображено;
* какой текст присутствует;
* в чём заключается шутка или абсурд;
* какие визуальные элементы создают юмор;
* есть ли очевидная культурная, интернетная или техническая отсылка.

Если на изображении есть какие то люди или персонажи, опиши их внешний вид, одежду, выражения лиц и позы. Если это известные личности, персонажи из фильмов, игр или мемов — укажи это.

Если смысл намеренно абсурдный или неочевидный, не пытайся искусственно придумывать объяснение. Опиши наблюдаемые элементы и предложи наиболее вероятную интерпретацию, обозначив её как предположение.

### Мемы и странный юмор

Будь готов к:

* shitpost'ам;
* намеренно тупым или бессмысленным мемам;
* surreal / absurdist humor;
* ironic humor;
* reaction images;
* cursed images;
* смешению текста и изображения;
* мемам, где шутка заключается именно в отсутствии смысла;
* локальным или нишевым интернет-мемам;
* изображениям, которые выглядят странно без контекста.
* interned humor
* dark humor
* Отсылкам на культурные (Казахстан, Россия, Гонконг, Китай, Америка), интернетные или технические явления.
* Отсылкам на недавние новости, аниме, видеоигры, фильмы, популярные сериалы, мемы и интернет-тренды.
* Отсылкам на киберпанк эстетику, киберпанк культуру, киберпанк игры, киберпанк фильмы и киберпанк литературу.

Не считай отсутствие очевидного смысла ошибкой изображения.

### Скриншоты

Особенно внимательно анализируй технические скриншоты.

Если это терминал:

* прочитай команды;
* прочитай вывод;
* выдели ошибки, warnings и важные сообщения;
* определи используемые программы, языки, библиотеки или окружение, если это видно;
* сохрани важные пути, имена файлов, URL, версии и другие конкретные данные.

Если это код:

* определи язык программирования;
* прочитай код настолько точно, насколько возможно;
* выдели очевидную ошибку или подозрительное место;
* объясни, что делает код, если это можно определить;
* не исправляй код самостоятельно, если пользователь этого не просил.

Если это ошибка или traceback:

* извлеки текст ошибки;
* укажи, где именно произошла ошибка;
* сохрани названия файлов, функций, строк и другие технические детали;
* не придумывай причину ошибки, если её нельзя определить по изображению.

Если это скриншот переписки:

* прочитай сообщения;
* укажи участников, если их имена видны;
* сохрани важный контекст разговора;
* различай текст сообщений, usernames, подписи интерфейса и другие элементы.

### Мат и грубый язык

Изображения могут содержать мат, оскорбления, сексуальные шутки, грубые выражения, черный юмор, и другой взрослый разговорный язык.

В контексте группового чата Neuromancer это **нормальная часть дружеского общения**. Не считай наличие мата само по себе причиной прекращать анализ изображения или возвращать пустой ответ.

Если мат или грубая формулировка важны для понимания шутки, переписки или контекста — передай их Neuromancer в нормальном виде. Не цензурируй слова без необходимости и не заменяй их расплывчатыми описаниями вроде «здесь используется нецензурная лексика».

При этом не приписывай изображению намерения или смысл, которых ты не можешь подтвердить.

### Точность

Не выдумывай то, чего нет на изображении.

Если текст плохо читается, изображение размыто или интерпретация неоднозначна — прямо скажи об этом.

Особенно внимательно сохраняй:

* имена;
* usernames;
* числа;
* даты;
* команды;
* код;
* URL;
* сообщения об ошибках;
* названия файлов;
* названия программ;
* технические термины.

Не нужно описывать каждую незначительную визуальную деталь. Передавай прежде всего информацию, которая поможет Neuromancer понять изображение и ответить на сообщение пользователя.

### Формат

Начни с краткого описания того, что изображено.

Затем добавь наиболее важные детали и контекст.

Если на изображении есть текст, приведи важный текст максимально точно.

Если это мем или шутка — объясни предполагаемый юмор.

Если это технический скриншот — выдели технически значимую информацию.

Если что-то невозможно уверенно определить, укажи это отдельно.

Отвечай на русском языке.
"""


class VisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class VisionResult:
    summary: str
    ocr: str
    raw_json: str


class OpenRouterVisionClient:
    def __init__(
        self,
        *,
        model: str,
        base_url: str = OPENROUTER_URL,
        timeout_seconds: float = 60.0,
    ):
        self.model = model
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds

    def analyze_image(self, path: Path) -> VisionResult:
        api_key = openrouter_api_key()
        if not api_key:
            raise VisionError("OPENROUTER_API_KEY is not set")

        mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        image_data = base64.b64encode(path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                            },
                        },
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 700,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/neuromancer-local",
                "X-Title": "Neuromancer Telegram Vision",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise VisionError(f"OpenRouter vision HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise VisionError(f"OpenRouter vision request failed: {exc}") from exc

        return parse_openrouter_response(raw_response)


def openrouter_api_key() -> str | None:
    for name in (
        "OPENROUTER_API_KEY",
        "OPENROUTER_KEY",
        "HERMES_OPENROUTER_API_KEY",
        "OPENROUTER_API_TOKEN",
    ):
        value = os.environ.get(name)
        if value:
            return value
    return None


def parse_openrouter_response(raw_response: str) -> VisionResult:
    try:
        data = json.loads(raw_response)
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise VisionError("OpenRouter vision response was not in the expected format") from exc

    content_text = content_to_text(content)
    parsed = parse_jsonish_content(content_text)
    if isinstance(parsed, dict):
        summary = str(parsed.get("summary") or "").strip()
        ocr = str(parsed.get("ocr") or "").strip()
        raw_json = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        return VisionResult(summary=summary or content_text.strip(), ocr=ocr, raw_json=raw_json)

    text = content_text.strip()
    return VisionResult(summary=text, ocr="", raw_json=json.dumps({"summary": text}, ensure_ascii=False))


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text")
                if value:
                    parts.append(str(value))
            elif item:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def parse_jsonish_content(content: str) -> Any:
    text = content.strip()
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
