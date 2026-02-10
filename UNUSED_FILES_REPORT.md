# Отчет: неиспользуемые и условно используемые файлы кода

Проверка выполнена по зависимостям импортов и точкам входа (run.py, app.py, Docker, web_bot).

---

## 1. Условно неиспользуемые основным приложением (run.py -> app.py)

Эти файлы **не участвуют** в работе основного сервера Voice Talker (FastAPI на порту 8300), но используются в других сценариях.

| Путь | Назначение | Когда используется |
|------|------------|--------------------|
| **src/main.py** | Альтернативная точка входа (CLI) | Только при явном запуске `python src/main.py` — цикл ввода пути к аудиофайлу, LLM, CRM, бот. |
| **web_bot/** | Отдельное веб-приложение | Только при запуске веб-бота (см. web_bot/README.md). |

---

## 2. Пакет tone: что не используется при инференсе ASR

Основное приложение использует только **инференс через ONNX**: `streaming_asr` -> `tone.pipeline` -> `tone.decoder`, `tone.logprob_splitter`, `tone.onnx_wrapper`. Остальное в `tone` не подключается при обычном запуске.

| Путь | Назначение | Используется |
|------|------------|--------------|
| **src/tone/demo/** | Демо-сайт и примеры для T-one ASR | Нет основным приложением. Запускается отдельно (демо). |
| **src/tone/__main__.py** | Запуск `python -m tone` | Нет основным приложением. Отдельная утилита. |
| **src/tone/scripts/export.py** | Экспорт модели в ONNX | Нет в рантайме. Только при экспорте модели. |
| **src/tone/training/** | Обучение модели (ToneForCTC, data_collator) | Нет при инференсе. Нужен только для обучения/экспорта. |
| **src/tone/nn/** | PyTorch-модель (Encoder, Conformer, feats и т.д.) | Нет при инференсе. Используется только в training и scripts/export. |
| **src/tone/project.py** | VERSION и метаданные | Используется в tone/__init__.py и tone/demo/website.py. При импорте только `tone.pipeline` не загружается. |

Итого по tone: при работе основного приложения загружаются только `tone.pipeline`, `tone.decoder`, `tone.logprob_splitter`, `tone.onnx_wrapper` и их зависимости. Модули `tone.nn`, `tone.training`, `tone.scripts`, `tone.demo` и `tone/__main__.py` в основном сценарии не используются.

---

## 3. Используемые файлы (для полноты картины)

- **run.py** — точка входа (uvicorn app:app).
- **run.bat** — запуск через `uvicorn` из папки `src`.
- **tts_microservice.py** — TTS-микросервис (Flask), используется в Dockerfile.tts.
- **check_tts_modules.py** — проверка модулей перед стартом TTS (Dockerfile.tts: `python check_tts_modules.py && python tts_microservice.py`).
- **src/app.py** — основное FastAPI-приложение.
- **src/config.py**, **src/database.py**, **src/llm.py** — используются app.py, main.py, web_bot.
- **src/streaming_asr.py** — используется app.py; тянет tone.pipeline.
- **src/tts_client.py**, **src/tts_transliteration_data.py** — используются app.py.
- **src/crm.py**, **src/whatsapp_telegram_bot.py**, **src/voice_processing.py**, **src/utils.py** — используются main.py и/или web_bot.

---

## 4. Рекомендации

- **Не удалять** перечисленные файлы без решения по сценариям: main.py и web_bot — альтернативные приложения; tone/demo, tone/scripts, tone/training, tone/nn — нужны для демо, экспорта и обучения.
- Если нужна **минимальная поставка** только для основного сервера (без демо и обучения), можно вынести в отдельный репозиторий или архив: tone/demo, tone/scripts, tone/training, tone/nn, tone/__main__.py, а также при необходимости main.py и web_bot — после явного решения не использовать эти сценарии.

---

*Отчет сгенерирован автоматически по результатам анализа импортов и точек входа.*
