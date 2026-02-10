"""Вспомогательные функции для загрузки аудио из байтов."""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave

import numpy as np
import numpy.typing as npt

from tone.onnx_wrapper import StreamingCTCModel


def _decode_with_miniaudio(raw_audio: bytes) -> npt.NDArray[np.int32] | None:
    """Попробовать декодировать аудио любового поддерживаемого формата через miniaudio."""
    try:
        import miniaudio
    except ImportError:
        return None

    with tempfile.NamedTemporaryFile(suffix=".tmp", delete=True) as temp_file:
        temp_file.write(raw_audio)
        temp_file.flush()
        decoded = miniaudio.decode_file(
            temp_file.name,
            nchannels=1,
            sample_rate=StreamingCTCModel.SAMPLE_RATE,
        )

    if decoded.sample_rate != StreamingCTCModel.SAMPLE_RATE or decoded.nchannels != 1:
        raise ValueError(
            "Аудио должно быть моно 8 кГц. Попробуйте прислать WAV 16-bit PCM "
            "или перекодировать файл перед отправкой.",
        )
    return np.asarray(decoded.samples, dtype=np.int16).astype(np.int32)


def _decode_with_ffmpeg(raw_audio: bytes) -> npt.NDArray[np.int32]:
    """Попробовать декодировать аудио через ffmpeg в 16-bit PCM 8 кГц моно."""
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(StreamingCTCModel.SAMPLE_RATE),
        "pipe:1",
    ]
    result = subprocess.run(cmd, input=raw_audio, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="ignore").strip()
        raise ValueError(f"ffmpeg decode failed: {stderr or 'unknown error'}")
    if not result.stdout:
        raise ValueError("ffmpeg returned empty audio")
    return np.frombuffer(result.stdout, dtype="<i2").astype(np.int32)


def _decode_pcm_wav(raw_audio: bytes) -> npt.NDArray[np.int32]:
    """Декодировать WAV 16-bit PCM 8 кГц, если miniaudio/ffmpeg недоступны."""
    with wave.open(io.BytesIO(raw_audio), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise ValueError("Ожидается моно WAV")
        if wav_file.getsampwidth() != 2:
            raise ValueError("Ожидается 16-bit PCM WAV")
        if wav_file.getframerate() != StreamingCTCModel.SAMPLE_RATE:
            raise ValueError("Ожидается частота дискретизации 8000 Гц")

        frames = wav_file.readframes(wav_file.getnframes())

    return np.frombuffer(frames, dtype="<i2").astype(np.int32)


def decode_audio_bytes(raw_audio: bytes) -> npt.NDArray[np.int32]:
    """Декодировать присланные байты аудио в формат, совместимый с пайплайном."""
    if len(raw_audio) == 0:
        raise ValueError("Получен пустой файл аудио")

    miniaudio_error: Exception | None = None
    ffmpeg_error: Exception | None = None

    try:
        decoded = _decode_with_miniaudio(raw_audio)
        if decoded is not None:
            return decoded
    except Exception as exc:  # noqa: BLE001
        miniaudio_error = exc

    try:
        return _decode_with_ffmpeg(raw_audio)
    except Exception as exc:  # noqa: BLE001
        ffmpeg_error = exc

    try:
        return _decode_pcm_wav(raw_audio)
    except Exception as exc:  # noqa: BLE001
        detail_parts = ["Не удалось декодировать аудио в формат WAV 16-bit моно 8 кГц."]
        if miniaudio_error is not None:
            detail_parts.append(f"miniaudio: {miniaudio_error}")
        if ffmpeg_error is not None:
            detail_parts.append(f"ffmpeg: {ffmpeg_error}")
        raise ValueError(" ".join(detail_parts)) from exc

