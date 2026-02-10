# Voice Talker - Dockerfile
# Голосовой помощник с потоковым ASR и локальной LLM


FROM python:3.11-slim

# Метаданные
LABEL maintainer="Voice Talker"
LABEL description="Voice assistant with streaming ASR and local LLM"
LABEL version="1.0.0"

# Установка системных зависимостей
# Примечание: portaudio19-dev нужен для сборки pyaudio
# cmake нужен для сборки kenlm
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    build-essential \
    portaudio19-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Создание рабочей директории
WORKDIR /app

# Копирование файлов зависимостей
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ ./src/
COPY data/ ./data/

# Примечание: модели ASR монтируются через volume или скачиваются из HuggingFace
# Создаем пустую директорию models
RUN mkdir -p /app/models

# Создание директории для данных (если не существует)
RUN mkdir -p /app/data

# Переменные окружения по умолчанию
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/src

# Настройки LLM (можно переопределить через docker-compose или -e)
ENV LLM_BASE_URL=http://host.docker.internal:1234/v1
ENV LLM_MODEL=google/gemma-3-4b
ENV LLM_MAX_TOKENS=2048
ENV LLM_TEMPERATURE=0.7

# Настройки ASR
# Примечание: если ASR_MODEL_PATH не задан, модель загружается из HuggingFace
# ENV ASR_MODEL_PATH=/app/models

# Настройки сервера
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8300

# Порт приложения
EXPOSE 8300

# Рабочая директория для запуска
WORKDIR /app/src

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8300/api/health')" || exit 1

# Команда запуска
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8300"]
