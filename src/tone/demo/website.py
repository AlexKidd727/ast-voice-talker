"""Module contain simple website implementation for demo purposes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable
import urllib.request

import numpy as np
import numpy.typing as npt
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from tone.demo.audio_utils import decode_audio_bytes
from tone.pipeline import StreamingCTCPipeline
from tone.python_terms import normalize_python_terms
from tone.project import VERSION

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)

_BYTES_PER_SAMPLE = 2


def _load_env_file(env_path: Path = Path(".env")) -> None:
    """Load values from .env without overriding already set variables."""
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def _as_bool(value: str | None, default: bool = False) -> bool:
    """Convert environment string to bool with sensible defaults."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_client_enable(value: str | None) -> bool | None:
    """Parse explicit client flag; returns None when not provided or invalid."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _log_task_exception(task: asyncio.Task[None]) -> None:
    """Log exceptions from background tasks to keep main flow clean."""
    try:
        task.result()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Фоновая отправка текста завершилась ошибкой: %s", exc)


def _schedule_background(coro: Awaitable[None]) -> None:
    """Fire-and-forget helper with exception logging for clarity."""
    task = asyncio.create_task(coro)
    task.add_done_callback(_log_task_exception)


@dataclass
class Settings:
    """Global website settings.

    Can be modified using environment variables.
    """

    cors_allow_all: bool = False
    load_from_folder: Path | None = field(default_factory=lambda: os.getenv("LOAD_FROM_FOLDER", None))
    enable_text_api: bool = field(default_factory=lambda: _as_bool(os.getenv("ENABLE_TEXT_API"), True))
    text_api_scheme: str = field(default_factory=lambda: os.getenv("TEXT_API_SCHEME", "http"))
    text_api_host: str = field(default_factory=lambda: os.getenv("TEXT_API_HOST", "host.docker.internal"))
    text_api_port: int = field(default_factory=lambda: int(os.getenv("TEXT_API_PORT", "8586")))
    text_api_url_override: str | None = field(default_factory=lambda: os.getenv("TEXT_API_URL"))
    text_api_timeout: float = field(default_factory=lambda: float(os.getenv("TEXT_API_TIMEOUT", "5.0")))

    @property
    def text_api_url(self) -> str:
        """Target URL for sending recognized text when the flag is on."""
        if self.text_api_url_override:
            return self.text_api_url_override.rstrip("/")
        return f"{self.text_api_scheme}://{self.text_api_host}:{self.text_api_port}/api/text"


class TextAPIClient:
    """Send recognized phrases to an external text API when enabled."""

    def __init__(self, settings: Settings) -> None:
        # Сохраняем параметры один раз, чтобы не дергать env внутри горячего пути.
        self.enabled = settings.enable_text_api
        self.url = settings.text_api_url
        self.timeout = settings.text_api_timeout

    async def send_phrase(self, phrase: str, *, enabled: bool | None = None) -> None:
        """Send a single phrase; silently skip if disabled or empty."""
        effective_enabled = self.enabled if enabled is None else (self.enabled and enabled)
        if not effective_enabled:
            return
        cleaned_phrase = phrase.strip()
        if not cleaned_phrase:
            return
        normalized = normalize_python_terms(cleaned_phrase)
        try:
            await asyncio.to_thread(self._post_text, normalized)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось отправить фразу в текстовый API: %s", exc)

    async def send_phrases(self, phrases: list[str], *, enabled: bool | None = None) -> None:
        """Send multiple phrases sequentially, preserving order."""
        effective_enabled = self.enabled if enabled is None else (self.enabled and enabled)
        if not effective_enabled:
            return
        tasks = [self.send_phrase(phrase, enabled=enabled) for phrase in phrases if phrase.strip()]
        if tasks:
            await asyncio.gather(*tasks)

    def _post_text(self, text: str) -> None:
        """Blocking sender run in thread to avoid slowing the event loop."""
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
            # читаем тело, чтобы корректно закрыть соединение
            response.read()


class SingletonPipeline:
    """Singleton object to store a single ASR pipeline."""

    pipeline: StreamingCTCPipeline | None = None

    def __new__(cls) -> None:
        """Ensure the class is never created."""
        raise RuntimeError("This is class is a singleton!")

    @classmethod
    def init(cls, settings: Settings) -> None:
        """Initialize singleton object using settings."""
        if settings.load_from_folder is None:
            cls.pipeline = StreamingCTCPipeline.from_hugging_face()
        else:
            cls.pipeline = StreamingCTCPipeline.from_local(settings.load_from_folder)

    @classmethod
    def process_chunk(
        cls,
        audio_chunk: StreamingCTCPipeline.InputType,
        state: StreamingCTCPipeline.StateType | None = None,
        *,
        is_last: bool = False,
    ) -> tuple[StreamingCTCPipeline.OutputType, StreamingCTCPipeline.StateType]:
        """Process audio chunk using ASR pipeline.

        See `StreamingCTCPipeline.forward` for more info.
        """
        if cls.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        return cls.pipeline.forward(audio_chunk, state, is_last=is_last)

    @classmethod
    def process_audio(cls, audio: StreamingCTCPipeline.InputType) -> StreamingCTCPipeline.OutputType:
        """Распознать полный аудиофайл (офлайн-режим)."""
        if cls.pipeline is None:
            raise RuntimeError("Pipeline is not initialized")
        return cls.pipeline.forward_offline(audio)


router = APIRouter()


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    """Простой healthcheck для мониторинга."""
    return {"status": "ok"}


@router.post("/transcribe")
async def transcribe_audio(request: Request, file: UploadFile = File(...)) -> dict[str, object]:
    """HTTP endpoint для офлайн-распознавания присланного аудиофайла."""
    raw_audio = await file.read()
    try:
        audio = decode_audio_bytes(raw_audio)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Не удалось обработать аудиофайл") from exc

    try:
        phrases = SingletonPipeline.process_audio(audio)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="Ошибка распознавания аудио") from exc

    text_api_client: TextAPIClient = request.app.state.text_api_client
    client_enable = _parse_client_enable(request.headers.get("x-enable-text-api"))
    await text_api_client.send_phrases([phrase.text for phrase in phrases], enabled=client_enable)

    response_phrases = [
        {"text": phrase.text, "start_time": phrase.start_time, "end_time": phrase.end_time}
        for phrase in phrases
    ]
    combined_text = " ".join(phrase["text"] for phrase in response_phrases).strip()
    return {"text": combined_text, "phrases": response_phrases}


async def get_chunk_stream(ws: WebSocket) -> AsyncIterator[tuple[npt.NDArray[np.int16], bool]]:
    """Get audio chunks from websocket and return them as async iterator."""
    audio_data = bytearray()
    # See description of PADDING in StreamingCTCPipeline
    audio_data.extend(np.zeros((StreamingCTCPipeline.PADDING,), dtype=np.int16).tobytes())

    is_last = False
    while True:
        await ws.send_json({"event": "ready"})
        recv_bytes = await ws.receive_bytes()
        if len(recv_bytes) == 0:  # Last chunk of audio
            is_last = True
            audio_data.extend(np.zeros((StreamingCTCPipeline.PADDING,), dtype=np.int16).tobytes())
            fill_chunk_size = -(len(audio_data) // _BYTES_PER_SAMPLE) % StreamingCTCPipeline.CHUNK_SIZE
            audio_data.extend(np.zeros((fill_chunk_size,), dtype=np.int16).tobytes())
        else:
            audio_data.extend(recv_bytes)

        while len(audio_data) >= StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE:
            chunk = np.frombuffer(audio_data[: StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE], dtype=np.int16)
            del audio_data[: StreamingCTCPipeline.CHUNK_SIZE * _BYTES_PER_SAMPLE]
            yield chunk, is_last and (len(audio_data) == 0)

        if len(recv_bytes) == 0:
            return


@router.websocket("/ws")
async def websocket_stt(ws: WebSocket) -> None:
    """Websocket endpoint for streaming audio processing."""
    await ws.accept()
    try:
        state: StreamingCTCPipeline.StateType | None = None
        text_api_client: TextAPIClient = ws.app.state.text_api_client
        client_enable = _parse_client_enable(ws.query_params.get("enable_text_api"))
        async for audio_chunk, is_last in get_chunk_stream(ws):
            output, state = SingletonPipeline.process_chunk(audio_chunk.astype(np.int32), state, is_last=is_last)
            for phrase in output:
                _schedule_background(text_api_client.send_phrase(phrase.text, enabled=client_enable))
                await ws.send_json(
                    {
                        "event": "transcript",
                        "phrase": {"text": phrase.text, "start_time": phrase.start_time, "end_time": phrase.end_time},
                    },
                )
    except WebSocketDisconnect:
        pass


def get_application() -> FastAPI:
    """Build and return FastAPI application."""
    _load_env_file()
    app = FastAPI(title="T-one Streaming ASR", version=VERSION, docs_url=None, redoc_url=None)
    settings = Settings()
    if settings.cors_allow_all:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.add_event_handler("startup", lambda: SingletonPipeline.init(settings))
    app.state.text_api_client = TextAPIClient(settings)

    app.include_router(router, prefix="/api")
    app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="Main website page")
    return app


app = get_application()
