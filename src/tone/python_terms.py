"""Справочник маппинга русских слов/произношений на Python-термины.

Фокус: распространённые библиотеки, типы данных и базовые сущности.
Операторы и управляющие конструкции сознательно не включены.
"""

from __future__ import annotations

import re
from typing import Mapping

# Базовые типы и встроенные контейнеры
PYTHON_TERM_MAP: dict[str, str] = {
    "лист": "list",
    "словарь": "dict",
    "кортеж": "tuple",
    "сет": "set",
    "множество": "set",
    "строка": "str",
    "стринг": "str",
    "инт": "int",
    "флоат": "float",
    "бул": "bool",
    "байтс": "bytes",
    "байт массив": "bytearray",
    "децимал": "Decimal",
    "дататайм": "datetime",
    "дейтайм": "datetime",
    "датакласс": "dataclass",
    "тайпхинт": "type hint",
    "аннотация": "annotation",
    "итератор": "iterator",
    "генератор": "generator",
}

# Стандартная библиотека и утилиты
PYTHON_TERM_MAP.update(
    {
        "айо": "asyncio",
        "айо файлы": "aiofiles",
        "айо хттп": "aiohttp",
        "логгинг": "logging",
        "итертулз": "itertools",
        "фанк тулз": "functools",
        "дейтютил": "dateutil",
        "таймдельта": "timedelta",
        "пазе": "pathlib",
        "патлиб": "pathlib",
        "сабпроцесс": "subprocess",
        "арк": "argparse",
        "датакласс": "dataclasses",
        "ипсило": "yaml",
        "жсон": "json",
        "пикл": "pickle",
        "тапл": "tuple",
        "кештулз": "cachetools",
        "лру кеш": "lru_cache",
    }
)

# Популярные сторонние библиотеки (данные, наука, веб, тестирование)
PYTHON_TERM_MAP.update(
    {
        # Данные / ML
        "нампай": "numpy",
        "нампи": "numpy",
        "пандас": "pandas",
        "сайпай": "scipy",
        "матплотлиб": "matplotlib",
        "сиборн": "seaborn",
        "пайторм": "pytorch",
        "пай торч": "pytorch",
        "тензор флоу": "tensorflow",
        "скитлерн": "scikit-learn",
        "хагинг фейс": "huggingface",
        "трансформерс": "transformers",
        "токенайзер": "tokenizers",
        "онн икс": "onnx",
        "онн икс рантайм": "onnxruntime",
        # Веб/ASGI/HTTP
        "фастапи": "FastAPI",
        "uvicorn": "uvicorn",
        "старлет": "starlette",
        "джанго": "Django",
        "фласк": "Flask",
        "саник": "Sanic",
        "кварк": "Quart",
        "айо хттп": "aiohttp",
        "реквестс": "requests",
        "http икс": "httpx",
        "вебсокет": "websockets",
        # CLI/инфраструктура
        "пип": "pip",
        "пип три": "pip",
        "поэтри": "poetry",
        "конда": "conda",
        "миниконда": "miniconda",
        "анаконда": "anaconda",
        "пип энв": "pipenv",
        "венв": "venv",
        # Тестирование и качество
        "пай тест": "pytest",
        "юнит тест": "unittest",
        "хипотезис": "hypothesis",
        "майпи": "mypy",
        "ров": "ruff",
        "флэйк эйт": "flake8",
        "блэк": "black",
        "исорт": "isort",
        # Сериализация / форматы
        "протобаф": "protobuf",
        "месседж пак": "msgpack",
        "паркет": "parquet",
        # БД / кеш
        "постгрес": "postgres",
        "постгрес кью эл": "postgresql",
        "постгре": "postgresql",
        "редис": "redis",
        "монго": "mongodb",
        "сиквел алхеми": "sqlalchemy",
        "эс кью эл алхеми": "sqlalchemy",
        "эс кью эль алхеми": "sqlalchemy",
        "алембик": "alembic",
        "пайдиэнтик": "pydantic",
        "пай дентек": "pydantic",
        "пи дентек": "pydantic",
        "пи дантик": "pydantic",
        # Сообщения / брокеры
        "раббит": "rabbitmq",
        "кафка": "kafka",
        "селери": "celery",
        # Разное
        "ткинтер": "tkinter",
        "пиоу ди эф": "pypdf",
        "пилоу": "pillow",
        "опенсви": "opencv",
    }
)


_TOKEN_SPLIT_RE = re.compile(r"([\\s,.;:()\\[\\]{}<>\"'`])")


def normalize_python_terms(text: str, mapping: Mapping[str, str] | None = None) -> str:
    """Заменить русские аналоги на целевые Python-термины.

    Разбивает по простым разделителям, сравнивает в lower-case, заменяет,
    сохраняя исходные разделители. Операторы и ключевые слова не маппятся,
    чтобы избежать ложных замен.
    """
    if not text:
        return text
    term_map = mapping or PYTHON_TERM_MAP
    parts = _TOKEN_SPLIT_RE.split(text)
    for i, part in enumerate(parts):
        key = part.strip().lower()
        if key and key in term_map:
            parts[i] = term_map[key]
    return "".join(parts)


__all__ = ["PYTHON_TERM_MAP", "normalize_python_terms"]

