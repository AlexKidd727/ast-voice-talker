# Voice Talker - Модуль потокового распознавания речи (ASR)
# Основан на наработках T-one для голосового ввода в реальном времени


from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Optional, Dict, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

# Константы для обработки аудио
_BYTES_PER_SAMPLE = 2
CHUNK_SAMPLES = 2400  # 300ms при 8kHz
CHUNK_BYTES = CHUNK_SAMPLES * 2


@dataclass
class ASRSettings:
    """
    Настройки ASR сервиса.
    Может быть инициализирован из конфигурации или переменных окружения.
    """
    # Путь к локальной модели (None = загрузка из HuggingFace)
    model_path: Optional[Path] = None
    
    # Настройки внешнего текстового API
    enable_text_api: bool = False
    text_api_url: str = "http://localhost:8586/api/text"
    text_api_timeout: float = 5.0
    
    # CORS настройки
    cors_allow_all: bool = True
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ASRSettings":
        """Создает настройки из словаря конфигурации."""
        asr_config = config.get('asr', {})
        
        model_path = asr_config.get('model_path')
        if model_path:
            model_path = Path(model_path)
        
        return cls(
            model_path=model_path,
            enable_text_api=asr_config.get('enable_text_api', False),
            text_api_url=asr_config.get('text_api_url', "http://localhost:8586/api/text"),
            text_api_timeout=asr_config.get('text_api_timeout', 5.0),
            cors_allow_all=asr_config.get('cors_allow_all', True)
        )


@dataclass
class TextPhrase:
    """
    Структура данных для фразы из ASR пайплайна.
    
    Attributes:
        text: Распознанный текст
        start_time: Время начала фразы (в секундах)
        end_time: Время окончания фразы (в секундах)
    """
    text: str
    start_time: float
    end_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует фразу в словарь."""
        return {
            "text": self.text,
            "start_time": self.start_time,
            "end_time": self.end_time
        }


def _log_task_exception(task: asyncio.Task[None]) -> None:
    """Логирует исключения из фоновых задач."""
    try:
        task.result()
    except Exception as exc:
        logger.warning("Фоновая отправка текста завершилась ошибкой: %s", exc)


def _schedule_background(coro: Awaitable[None]) -> None:
    """Fire-and-forget хелпер с логированием исключений."""
    task = asyncio.create_task(coro)
    task.add_done_callback(_log_task_exception)


class TextAPIClient:
    """
    Клиент для отправки распознанных фраз во внешний текстовый API.
    Используется для интеграции с другими системами (например, с LLM).
    """
    
    def __init__(self, settings: ASRSettings) -> None:
        self.enabled = settings.enable_text_api
        self.url = settings.text_api_url
        self.timeout = settings.text_api_timeout
    
    async def send_phrase(self, phrase: str, *, enabled: Optional[bool] = None) -> None:
        """Отправляет одну фразу; пропускает если отключено или пусто."""
        effective_enabled = self.enabled if enabled is None else (self.enabled and enabled)
        if not effective_enabled:
            return
        
        cleaned_phrase = phrase.strip()
        if not cleaned_phrase:
            return
        
        try:
            await asyncio.to_thread(self._post_text, cleaned_phrase)
        except Exception as exc:
            logger.warning("Не удалось отправить фразу в текстовый API: %s", exc)
    
    async def send_phrases(self, phrases: list[str], *, enabled: Optional[bool] = None) -> None:
        """Отправляет несколько фраз последовательно."""
        effective_enabled = self.enabled if enabled is None else (self.enabled and enabled)
        if not effective_enabled:
            return
        
        tasks = [self.send_phrase(phrase, enabled=enabled) for phrase in phrases if phrase.strip()]
        if tasks:
            await asyncio.gather(*tasks)
    
    def _post_text(self, text: str) -> None:
        """Блокирующий отправщик, выполняется в отдельном потоке."""
        payload = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            if response.status >= 300:
                raise RuntimeError(f"Text API returned status {response.status}")
            response.read()


class StreamingASRPipeline:
    """
    Потоковый ASR пайплайн для распознавания речи в реальном времени.
    
    Этот класс является оберткой над T-one StreamingCTCPipeline и предоставляет
    простой интерфейс для интеграции с веб-приложением.
    """
    
    # Константы из T-one
    PADDING: int = 2400  # 300ms * 8KHz
    CHUNK_SIZE: int = 2400  # Размер чанка в семплах (300ms при 8kHz)
    SAMPLE_RATE: int = 8000
    
    _pipeline = None  # Singleton для пайплайна
    _initialized = False
    
    def __init__(self, settings: Optional[ASRSettings] = None):
        """
        Инициализация ASR пайплайна.
        
        Args:
            settings: Настройки ASR (опционально)
        """
        self.settings = settings or ASRSettings()
        self._state = None
    
    @classmethod
    def init_pipeline(cls, settings: Optional[ASRSettings] = None) -> bool:
        """
        Инициализирует singleton пайплайн.
        
        Args:
            settings: Настройки ASR
            
        Returns:
            True если инициализация успешна
        """
        if cls._initialized:
            return True
        
        try:
            # Примечание: Импортируем T-one только при инициализации
            # чтобы избежать ошибок если библиотека не установлена
            from tone.pipeline import StreamingCTCPipeline
            
            settings = settings or ASRSettings()
            
            if settings.model_path:
                cls._pipeline = StreamingCTCPipeline.from_local(settings.model_path)
                logger.info(f"ASR пайплайн загружен из локальной папки: {settings.model_path}")
            else:
                cls._pipeline = StreamingCTCPipeline.from_hugging_face()
                logger.info("ASR пайплайн загружен из HuggingFace")
            
            cls._initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Не удалось импортировать T-one: {e}")
            logger.error("Установите T-one: pip install tone-asr или скопируйте модуль tone")
            return False
        except Exception as e:
            logger.error(f"Ошибка инициализации ASR пайплайна: {e}")
            return False
    
    @classmethod
    def is_initialized(cls) -> bool:
        """Проверяет, инициализирован ли пайплайн."""
        return cls._initialized
    
    def process_chunk(
        self,
        audio_chunk: npt.NDArray[np.int32],
        is_last: bool = False
    ) -> list[TextPhrase]:
        """
        Обрабатывает чанк аудио и возвращает распознанные фразы.
        
        Args:
            audio_chunk: Чанк аудио (2400 семплов = 300ms)
            is_last: Флаг последнего чанка
            
        Returns:
            Список распознанных фраз
        """
        if not self._initialized or self._pipeline is None:
            logger.error("ASR пайплайн не инициализирован")
            return []
        
        try:
            output, self._state = self._pipeline.forward(
                audio_chunk, 
                self._state, 
                is_last=is_last
            )
            
            return [
                TextPhrase(
                    text=phrase.text,
                    start_time=phrase.start_time,
                    end_time=phrase.end_time
                )
                for phrase in output
            ]
            
        except Exception as e:
            logger.error(f"Ошибка обработки аудио чанка: {e}")
            return []
    
    def process_audio(self, audio: npt.NDArray[np.int32]) -> list[TextPhrase]:
        """
        Обрабатывает полный аудиофайл (офлайн-режим).
        
        Args:
            audio: Полный аудиофайл в формате numpy array
            
        Returns:
            Список распознанных фраз
        """
        if not self._initialized or self._pipeline is None:
            logger.error("ASR пайплайн не инициализирован")
            return []
        
        try:
            output = self._pipeline.forward_offline(audio)
            
            return [
                TextPhrase(
                    text=phrase.text,
                    start_time=phrase.start_time,
                    end_time=phrase.end_time
                )
                for phrase in output
            ]
            
        except Exception as e:
            logger.error(f"Ошибка обработки аудио: {e}")
            return []
    
    def reset_state(self) -> None:
        """Сбрасывает состояние пайплайна для новой сессии."""
        self._state = None
    
    def finalize(self) -> list[TextPhrase]:
        """
        Завершает обработку и возвращает оставшиеся фразы.
        
        Returns:
            Список оставшихся распознанных фраз
        """
        if not self._initialized or self._pipeline is None:
            return []
        
        try:
            output, self._state = self._pipeline.finalize(self._state)
            
            return [
                TextPhrase(
                    text=phrase.text,
                    start_time=phrase.start_time,
                    end_time=phrase.end_time
                )
                for phrase in output
            ]
            
        except Exception as e:
            logger.error(f"Ошибка финализации: {e}")
            return []


async def get_chunk_stream(ws, pipeline: StreamingASRPipeline):
    """
    Генератор чанков аудио из WebSocket.
    
    Args:
        ws: WebSocket соединение
        pipeline: ASR пайплайн
        
    Yields:
        Кортеж (audio_chunk, is_last)
    """
    audio_data = bytearray()
    # Добавляем начальный паддинг
    audio_data.extend(np.zeros((pipeline.PADDING,), dtype=np.int16).tobytes())
    
    is_last = False
    while True:
        await ws.send_json({"event": "ready"})
        recv_bytes = await ws.receive_bytes()
        
        if len(recv_bytes) == 0:  # Последний чанк
            is_last = True
            audio_data.extend(np.zeros((pipeline.PADDING,), dtype=np.int16).tobytes())
            fill_chunk_size = -(len(audio_data) // _BYTES_PER_SAMPLE) % pipeline.CHUNK_SIZE
            audio_data.extend(np.zeros((fill_chunk_size,), dtype=np.int16).tobytes())
        else:
            audio_data.extend(recv_bytes)
        
        while len(audio_data) >= pipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE:
            chunk = np.frombuffer(
                audio_data[:pipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE], 
                dtype=np.int16
            )
            del audio_data[:pipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE]
            yield chunk.astype(np.int32), is_last and (len(audio_data) == 0)
        
        if len(recv_bytes) == 0:
            return


def decode_audio_bytes(raw_audio: bytes) -> npt.NDArray[np.int32]:
    """
    Декодирует байты аудио в формат, совместимый с пайплайном.
    Поддерживает форматы: WAV (PCM 16-bit), и через ffmpeg любые другие.
    
    Args:
        raw_audio: Сырые байты аудио
        
    Returns:
        numpy array с аудио данными
        
    Raises:
        ValueError: Если не удалось декодировать аудио
    """
    import io
    import wave
    import subprocess
    
    if len(raw_audio) == 0:
        raise ValueError("Получен пустой файл аудио")
    
    SAMPLE_RATE = 8000
    
    # Попытка декодировать как WAV
    try:
        with wave.open(io.BytesIO(raw_audio), "rb") as wav_file:
            if wav_file.getnchannels() == 1 and wav_file.getsampwidth() == 2:
                if wav_file.getframerate() == SAMPLE_RATE:
                    frames = wav_file.readframes(wav_file.getnframes())
                    return np.frombuffer(frames, dtype="<i2").astype(np.int32)
    except Exception:
        pass
    
    # Попытка декодировать через ffmpeg
    try:
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "s16le",
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "pipe:1",
        ]
        result = subprocess.run(cmd, input=raw_audio, capture_output=True)
        if result.returncode == 0 and result.stdout:
            return np.frombuffer(result.stdout, dtype="<i2").astype(np.int32)
    except Exception:
        pass
    
    raise ValueError(
        "Не удалось декодировать аудио. "
        "Поддерживаемые форматы: WAV 16-bit моно 8 кГц, "
        "или любой формат при наличии ffmpeg."
    )
