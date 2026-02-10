# Voice Talker - Голосовой помощник с AI

Комплексная система для голосового ввода в реальном времени с обработкой через локальную LLM модель (Gemma 3b или другие).

## Возможности

### Потоковое распознавание речи (ASR)
- Распознавание речи в реальном времени на основе T-one
- WebSocket API для потокового ввода
- Поддержка микрофона в браузере
- Загрузка и распознавание аудиофайлов

### Локальная LLM модель
- Интеграция с локальным LLM сервером (LM Studio)
- Модель: google/gemma-3-4b
- OpenAI-совместимый API (OpenAI SDK 2.x)
- Потоковая генерация ответов

## Быстрый старт

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Настройка LLM сервера

Убедитесь, что LM Studio запущен с моделью `google/gemma-3-4b`. В примерах ниже указан адрес `192.168.1.250` — **замените его на IP вашего хоста** (компьютера, где запущен LM Studio).  
- **Локальный запуск** (Python, run.bat): если LM Studio на том же компьютере — используйте `127.0.0.1`.  
- **Запуск через Docker**: `127.0.0.1` не подойдёт (это адрес контейнера). Укажите IP хоста (например, `192.168.1.250`) или `host.docker.internal` (Docker Desktop).

При необходимости измените настройки в `data/config.json`:

```json
{
  "llm": {
    "api_key": "not-needed",
    "model_name": "google/gemma-3-4b",
    "base_url": "http://192.168.1.250:1234/v1",
    "max_tokens": 2048,
    "temperature": 0.7
  }
}
```

В `base_url` замените `192.168.1.250` на адрес вашего хоста. Для локального запуска — `127.0.0.1`; для Docker — IP хоста или `host.docker.internal` (не `127.0.0.1`).

### 3. Запуск сервера

**Windows:**
```bash
run.bat
```

**Python:**
```bash
python run.py
```

**Или напрямую:**
```bash
cd src
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Открытие интерфейса

Откройте в браузере: http://localhost:8300

## Файлы моделей

Файлы моделей в репозиторий не входят. Их нужно скачать и положить в папку `models/` в корне проекта.

### Структура папки models

```
models/
├── t-one/                    # ASR (распознавание речи)
│   ├── model.onnx
│   └── kenlm.bin
├── v3_1_ru.pt                # Silero TTS (опционально)
├── v5_ru.pt                  # Silero TTS
└── v5_cis_base_nostress.pt   # Silero TTS (опционально)
```

### ASR (T-one)

Нужны для потокового распознавания речи. Оба файла — в подпапку `models/t-one/`.

| Файл        | Откуда скачать |
|-------------|----------------|
| model.onnx  | [Hugging Face: t-tech/T-one](https://huggingface.co/t-tech/T-one) — файл `model.onnx` |
| kenlm.bin   | [Hugging Face: t-tech/T-one](https://huggingface.co/t-tech/T-one) — файл `kenlm.bin` |

Страница репозитория: https://huggingface.co/t-tech/T-one (кнопка «Files and versions», скачать нужные файлы). Положите их в папку `models/t-one/`.

Чтобы бэкенд использовал локальные модели, в `data/config.json` в секции `asr` укажите путь к папке (относительно корня проекта или абсолютный), например:

```json
"asr": {
  "model_path": "models/t-one",
  ...
}
```

Если `model_path` не указан или `null`, пайплайн при первом запуске попытается скачать артефакты с Hugging Face в кэш.

### TTS (Silero)

Нужны для синтеза речи (TTS-микросервис). Файлы класть в корень папки `models/` (рядом с папкой `t-one`).

| Файл                     | Описание              | Откуда скачать |
|--------------------------|-----------------------|----------------|
| v5_ru.pt                 | V5, основные голоса   | https://models.silero.ai/models/tts/ru/v5_ru.pt |
| v3_1_ru.pt               | V3.1 (опционально)    | https://models.silero.ai/models/tts/ru/v3_1_ru.pt |
| v5_cis_base_nostress.pt  | V5 CIS, новые голоса  | https://models.silero.ai/models/tts/ru/v5_cis_base_nostress.pt |

Достаточно хотя бы одной модели (обычно `v5_ru.pt`). Если файлов нет в `models/`, микросервис при первом запросе скачает выбранную модель в кэш PyTorch.

## Архитектура

```
voice_talker/
├── src/                      # Основные модули бэкенда
│   ├── app.py               # FastAPI сервер
│   ├── streaming_asr.py     # Потоковый ASR
│   ├── llm.py               # LLM клиент
│   ├── tts_client.py        # TTS клиент
│   ├── config.py            # Конфигурация
│   ├── static/              # Статические файлы
│   │   └── index.html       # Веб-интерфейс
│   ├── database.py          # Работа с БД
│   ├── voice_processing.py  # Обработка голоса
│   ├── crm.py               # CRM система
│   └── utils.py             # Утилиты
├── data/
│   └── config.json          # Настройки
├── run.py                   # Скрипт запуска
├── run.bat                  # Скрипт запуска Windows
└── requirements.txt         # Зависимости Python
```

## API Endpoints

### REST API

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/api/health` | Проверка статуса сервера |
| POST | `/api/transcribe` | Распознавание аудиофайла |
| POST | `/api/chat` | Отправка сообщения в LLM |
| POST | `/api/chat/stream` | Потоковый чат с LLM (SSE) |
| POST | `/api/analyze` | Анализ текста через LLM |
| GET | `/api/llm/test` | Тест соединения с LLM |

### WebSocket API

| Endpoint | Описание |
|----------|----------|
| `/api/ws` | Потоковое распознавание речи |
| `/api/ws/chat` | Распознавание + ответ LLM + TTS |

### WebSocket протокол

**Клиент -> Сервер:**
- Бинарные данные: PCM 16-bit 8kHz mono чанки
- Пустой пакет: сигнал завершения

**Сервер -> Клиент:**
```json
{"event": "ready"}           // Готов принять данные
{"event": "transcript", "phrase": {"text": "...", "start_time": 0.0, "end_time": 1.5}}
{"event": "llm_response", "text": "..."}  // Только для /ws/chat
```

## Конфигурация

### Переменные окружения

| Переменная | Описание | Значение по умолчанию |
|------------|----------|----------------------|
| `LLM_BASE_URL` | URL LLM сервера (замените `192.168.1.250` на адрес вашего хоста) | `http://192.168.1.250:1234/v1` |
| `LLM_MODEL` | Название модели | `google/gemma-3-4b` |
| `LLM_MAX_TOKENS` | Макс. токенов | `2048` |
| `LLM_TEMPERATURE` | Температура | `0.7` |
| `SERVER_HOST` | Хост сервера | `0.0.0.0` |
| `SERVER_PORT` | Порт сервера | `8300` |

### data/config.json

В `llm.base_url` замените `192.168.1.250` на IP вашего хоста. Локальный запуск — `127.0.0.1`; при запуске через Docker используйте IP хоста или `host.docker.internal`, так как `127.0.0.1` из контейнера указывает на сам контейнер.

```json
{
  "llm": {
    "api_key": "not-needed",
    "model_name": "google/gemma-3-4b",
    "base_url": "http://192.168.1.250:1234/v1",
    "max_tokens": 2048,
    "temperature": 0.7
  },
  "asr": {
    "model_path": null,
    "sample_rate": 8000,
    "enable_text_api": false
  },
  "server": {
    "host": "0.0.0.0",
    "port": 8300
  },
  "tts": {
    "url": "http://tts-service:5002",
    "speaker": "kseniya"
  }
}
```

## Технологии

### Бэкенд (Python)
- **Python 3.9+** - основной язык
- **FastAPI** - веб-фреймворк
- **OpenAI SDK 2.x** - клиент для LLM
- **T-one** - потоковое распознавание речи
- **Silero TTS** - синтез речи
- **WebSocket** - реальное время
- **Bootstrap 5** - UI компоненты

## Требования
- Python 3.9+
- LM Studio (или другой OpenAI-совместимый сервер)
- ffmpeg (опционально, для декодирования аудио)
- Микрофон (для записи в браузере)

## Решение проблем

### LLM не отвечает

1. Проверьте, что LM Studio запущен
2. Проверьте URL в конфигурации
3. Проверьте, что модель загружена

### ASR не работает

1. Убедитесь, что в папке `models/t-one/` лежат файлы `model.onnx` и `kenlm.bin` (см. раздел «Файлы моделей»).
2. В `data/config.json` в секции `asr` укажите `"model_path": "models/t-one"` для использования локальных моделей.

### Микрофон не работает в браузере

1. Используйте HTTPS или localhost
2. Разрешите доступ к микрофону в браузере

## Лицензия

MIT License
