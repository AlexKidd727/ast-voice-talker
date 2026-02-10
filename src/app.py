# Voice Talker - Главный сервер приложения
# Комбинирует голосовой ввод в реальном времени (ASR) и обработку через LLM
# Основан на наработках T-one для потокового ASR


from __future__ import annotations

import asyncio
import json
import logging
import urllib.parse
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse

from config import load_config
from llm import LLM
from database import (
    create_connection, 
    create_tables, 
    init_default_roles,
    init_default_settings,
    get_all_roles,
    get_role_by_id,
    create_role,
    update_role,
    delete_role,
    get_setting,
    get_all_settings,
    set_setting,
    get_llm_settings,
    update_llm_settings
)
from streaming_asr import (
    ASRSettings, 
    StreamingASRPipeline, 
    TextAPIClient, 
    TextPhrase,
    decode_audio_bytes,
    get_chunk_stream,
    _schedule_background,
    CHUNK_BYTES
)
from tts_client import TTSClient, get_tts_client, initialize_tts_client

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Версия приложения
VERSION = "1.0.0"

# Загрузка конфигурации
config = load_config()

# Инициализация LLM
llm_client = LLM(config)

# Инициализация базы данных
# Примечание: путь к БД относительно data директории для персистентности в Docker
DB_PATH = Path(__file__).parent.parent / "data" / "voice_talker.db"
db_conn = create_connection(str(DB_PATH))
if db_conn:
    create_tables(db_conn)
    init_default_roles(db_conn)
    init_default_settings(db_conn)
    logger.info(f"База данных инициализирована: {DB_PATH}")
    
    # Загружаем сохраненную модель из БД
    # Примечание: применяем к LLM клиенту модель, выбранную пользователем ранее
    saved_model = get_setting(db_conn, "llm_model")
    if saved_model and llm_client.is_available():
        llm_client.set_model(saved_model)
        logger.info(f"Загружена модель из БД: {saved_model}")
else:
    logger.error("Не удалось подключиться к базе данных")

# Создание роутера API
router = APIRouter()


def _parse_client_enable(value: Optional[str]) -> Optional[bool]:
    """Парсит флаг клиента; возвращает None если не указан."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return None


import re

def _strip_code_for_tts(text: str) -> str:
    """
    Удаляет блоки кода, размышлений и Markdown разметку из текста перед озвучиванием TTS.
    Сохраняет абзацы и новые строки для естественного озвучивания.
    
    Удаляет:
    - Блоки размышлений <think>...</think>
    - Специальные токены LLM <|begin_of_box|>...<|end_of_box|>
    - Многострочные блоки кода ```...```
    - Инлайн код `...`
    - Markdown разметку (заголовки, списки, жирный/курсив, ссылки, изображения, цитаты)
    - Заменяет на фразу "смотри код на экране" если был код
    """
    original_text = text
    
    # Удаляем блоки размышлений <think>...</think>
    # Примечание: некоторые LLM выдают размышления в таких тегах
    # Важно: удаляем полностью, включая все содержимое, даже если оно многострочное
    # Используем жадный квантификатор и флаги для многострочного поиска
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
    
    # Также удаляем блоки с экранированными тегами (на случай, если они были экранированы)
    text = re.sub(r'&lt;think&gt;.*?&lt;/think&gt;', '', text, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
    
    # Удаляем незакрытые блоки <think> (на случай, если закрывающий тег отсутствует)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'&lt;think&gt;.*', '', text, flags=re.DOTALL | re.IGNORECASE | re.MULTILINE)
    
    # Удаляем специальные токены LLM <|begin_of_box|>...<|end_of_box|>
    # Примечание: извлекаем содержимое, убирая только теги
    text = re.sub(r'<\|begin_of_box\|>(.*?)<\|end_of_box\|>', r'\1', text, flags=re.DOTALL)
    
    # Удаляем специальные токены LLM вида <|token_name|>content
    # Примечание: удаляем токены типа <|channel|>, <|constrain|>, <|message|>
    # Пример: <|channel|>commentary to=developer <|constrain|>response<|message|>текст
    # Результат: текст (оставляем только текст после последнего токена <|message|>)
    
    # Если есть токен <|message|>, оставляем только текст после него
    message_match = re.search(r'<\|message\|>(.*)', text, re.DOTALL)
    if message_match:
        # Оставляем только текст после <|message|>
        text = message_match.group(1)
    else:
        # Если нет <|message|>, удаляем все токены и их содержимое
        # Удаляем все токены и их содержимое до следующего токена
        text = re.sub(r'<\|[^|]+\|>[^<]*?(?=<\||$)', '', text, flags=re.MULTILINE)
        # Удаляем оставшиеся одиночные токены без содержимого
        text = re.sub(r'<\|[^|]+\|>', '', text)
    
    # Удаляем многострочные блоки кода ```язык\n...\n```
    # Примечание: используем re.DOTALL чтобы . включал переносы строк
    text = re.sub(r'```[\w]*\n.*?```', '', text, flags=re.DOTALL)
    
    # Удаляем простые многострочные блоки ``` без указания языка
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    
    # Удаляем инлайн код `...`
    text = re.sub(r'`[^`]+`', '', text)
    
    # Удаляем Markdown изображения ![alt](url) или ![alt](url "title")
    text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', text)
    
    # Удаляем Markdown ссылки [text](url) или [text](url "title"), оставляя только текст
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    
    # Удаляем Markdown ссылки-ссылки [text][ref], оставляя только текст
    text = re.sub(r'\[([^\]]+)\]\[[^\]]+\]', r'\1', text)
    
    # Удаляем автоссылки <url>, оставляя только url
    text = re.sub(r'<([^>]+)>', r'\1', text)
    
    # Удаляем Markdown жирный текст **text** или __text__, оставляя только текст
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    
    # Удаляем Markdown курсив *text* или _text_, оставляя только текст
    # Примечание: делаем это после жирного, чтобы не конфликтовать
    text = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'\1', text)
    text = re.sub(r'(?<!_)_([^_]+)_(?!_)', r'\1', text)
    
    # Удаляем Markdown зачеркнутый текст ~~text~~
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    
    # Удаляем Markdown заголовки (# ## ###), оставляя текст
    # Примечание: убираем решетки в начале строки, чтобы не озвучивались
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Удаляем Markdown заголовки подчеркиванием (=== или ---)
    text = re.sub(r'^={3,}$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^-{3,}$', '', text, flags=re.MULTILINE)
    
    # Убираем Markdown списки (-, *, +, 1., 2. и т.д.), оставляя текст
    # Примечание: убираем маркеры списков, но сохраняем переносы строк
    text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Убираем горизонтальные линии Markdown (--- или *** или ___)
    text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
    
    # Удаляем Markdown блоки цитат (>), оставляя текст
    text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
    
    # Удаляем Markdown таблицы (уже обрабатываются в _preprocess_text, но на всякий случай)
    # Примечание: таблицы должны обрабатываться до этого этапа
    
    # Удаляем Markdown код в блоках (уже обработано выше)
    
    # Удаляем Markdown фрагменты кода (уже обработано выше)
    
    # Нормализуем переносы строк: сохраняем абзацы (двойные переносы), но убираем лишние
    # Примечание: заменяем 3+ переноса на 2 (абзац), сохраняем одиночные переносы
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Убираем лишние пробелы в начале и конце строк, но сохраняем структуру
    lines = text.split('\n')
    cleaned_lines = [line.strip() for line in lines]
    text = '\n'.join(cleaned_lines)
    
    # Убираем множественные пробелы внутри строк (но сохраняем одиночные)
    text = re.sub(r' {2,}', ' ', text)
    
    # Убираем пустые строки в начале и конце
    text = text.strip()
    
    # Если был удален код, добавляем примечание
    if len(text) < len(original_text) * 0.7 and len(original_text) > 50:
        # Был удален значительный объем (код), добавляем примечание
        if text:
            text = text + " Подробности смотри в коде на экране."
        else:
            text = "Смотри код на экране."
    
    return text if text else "Смотри ответ на экране."


def _is_meaningful_message(text: str) -> bool:
    """
    Проверяет, является ли сообщение значимым для отправки в LLM.
    
    Фильтрует:
    - Пустые сообщения
    - Сообщения, содержащие только буквы и короче 3 символов (кроме "да" и "нет")
    
    Args:
        text: Текст сообщения
        
    Returns:
        True если сообщение значимое и должно быть отправлено в LLM, False иначе
    """
    if not text or not text.strip():
        return False
    
    # Убираем пробелы для проверки
    text_clean = text.strip()
    
    # Разрешаем "да" и "нет" в любом регистре
    if text_clean.lower() in ['да', 'нет']:
        return True
    
    # Если сообщение содержит только буквы (без цифр, знаков препинания)
    # и короче 3 символов - не отправляем
    if len(text_clean) < 3 and re.match(r'^[а-яёa-z]+$', text_clean, re.IGNORECASE):
        return False
    
    # Все остальные сообщения считаем значимыми
    return True


# ==================== API Endpoints ====================

@router.get("/health")
async def healthcheck() -> dict:
    """Проверка работоспособности сервиса."""
    return {
        "status": "ok",
        "version": VERSION,
        "asr_initialized": StreamingASRPipeline.is_initialized(),
        "llm_available": llm_client.is_available()
    }


@router.post("/transcribe")
async def transcribe_audio(request: Request, file: UploadFile = File(...)) -> dict:
    """
    HTTP endpoint для офлайн-распознавания аудиофайла.
    
    Поддерживаемые форматы: WAV (16-bit PCM 8kHz mono), или любой через ffmpeg.
    """
    raw_audio = await file.read()
    
    try:
        audio = decode_audio_bytes(raw_audio)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Не удалось обработать аудиофайл") from exc
    
    # Создаем пайплайн для обработки
    pipeline = StreamingASRPipeline()
    
    try:
        phrases = pipeline.process_audio(audio)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Ошибка распознавания аудио") from exc
    
    # Отправка в текстовый API если включено
    text_api_client: TextAPIClient = request.app.state.text_api_client
    client_enable = _parse_client_enable(request.headers.get("x-enable-text-api"))
    await text_api_client.send_phrases([phrase.text for phrase in phrases], enabled=client_enable)
    
    response_phrases = [phrase.to_dict() for phrase in phrases]
    combined_text = " ".join(phrase["text"] for phrase in response_phrases).strip()
    
    return {"text": combined_text, "phrases": response_phrases}


@router.websocket("/ws")
async def websocket_stt(ws: WebSocket) -> None:
    """
    WebSocket endpoint для потокового распознавания речи.
    
    Протокол:
    1. Клиент получает {"event": "ready"} когда сервер готов принять данные
    2. Клиент отправляет бинарные данные (PCM 16-bit 8kHz mono)
    3. Сервер отправляет {"event": "transcript", "phrase": {...}} для каждой распознанной фразы
    4. Клиент отправляет пустой пакет для завершения
    """
    await ws.accept()
    
    try:
        pipeline = StreamingASRPipeline()
        text_api_client: TextAPIClient = ws.app.state.text_api_client
        client_enable = _parse_client_enable(ws.query_params.get("enable_text_api"))
        
        async for audio_chunk, is_last in get_chunk_stream(ws, pipeline):
            phrases = pipeline.process_chunk(audio_chunk, is_last=is_last)
            
            for phrase in phrases:
                # Отправка в текстовый API в фоне
                _schedule_background(text_api_client.send_phrase(phrase.text, enabled=client_enable))
                
                # Отправка результата клиенту
                await ws.send_json({
                    "event": "transcript",
                    "phrase": phrase.to_dict()
                })
                
    except WebSocketDisconnect:
        logger.info("WebSocket отключен клиентом")
    except Exception as e:
        logger.error(f"Ошибка WebSocket: {e}")


@router.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket) -> None:
    """
    WebSocket endpoint для чата с LLM на основе распознанной речи.
    
    Протокол:
    1. Работает как /ws для распознавания речи
    2. После каждой распознанной фразы отправляет запрос в LLM
    3. Ответ LLM отправляется клиенту
    4. Если TTS доступен, также отправляется аудио ответа
    
    Query params:
    - speaker: голос TTS (kseniya, xenia, baya, aidar, eugene)
    - system_prompt: кастомный системный промт (URL-encoded)
    """
    await ws.accept()
    
    if not llm_client.is_available():
        await ws.send_json({
            "event": "error",
            "message": "LLM недоступен"
        })
        await ws.close()
        return
    
    try:
        pipeline = StreamingASRPipeline()
        # Примечание: получаем историю сообщений из query params, если передана
        # Если история передана, используем её, иначе создаем новую пустую
        history_param = ws.query_params.get("history", "")
        if history_param:
            # Примечание: декодируем историю из URL-encoded формата
            history_text = urllib.parse.unquote(history_param)
            # Разбиваем на строки и формируем список истории
            message_history = [line.strip() for line in history_text.split('\n') if line.strip()]
        else:
            message_history = []
        MAX_HISTORY = 6  # Храним 6 последних обменов (12 сообщений)
        tts_client = get_tts_client()
        tts_available = await tts_client.is_ready()
        
        # Примечание: получаем голос, системный промт и max_tokens из query params
        speaker = ws.query_params.get("speaker", "kseniya")
        system_prompt = ws.query_params.get("system_prompt", "")
        max_tokens = int(ws.query_params.get("max_tokens", "2048"))
        
        # Маппинг голосов на имена и пол
        # kseniya -> Маша, xenia -> Женя, baya -> Юля, aidar -> Дима, eugene -> Евгений
        speaker_config = {
            "kseniya": {"name": "Маша", "gender": "female"},
            "xenia": {"name": "Женя", "gender": "female"},
            "baya": {"name": "Юля", "gender": "female"},
            "aidar": {"name": "Дима", "gender": "male"},
            "eugene": {"name": "Евгений", "gender": "male"}
        }
        config_data = speaker_config.get(speaker, {"name": "Маша", "gender": "female"})
        gender = config_data["gender"]
        assistant_name = config_data["name"]
        
        # Примечание: инициализируем флаги остановки
        if not hasattr(ws.app.state, 'stop_flags'):
            ws.app.state.stop_flags = {}
        stop_flag_key = id(ws)
        ws.app.state.stop_flags[stop_flag_key] = False
        
        async for audio_chunk, is_last in get_chunk_stream(ws, pipeline):
            phrases = pipeline.process_chunk(audio_chunk, is_last=is_last)
            
            for phrase in phrases:
                recognized_text_lower = phrase.text.lower().strip()
                
                # Примечание: проверяем команды остановки воспроизведения (одно слово)
                # Команды: стоп, достаточно, хватит, понятно
                # Примечание: проверяем, содержит ли текст одно из командных слов как отдельное слово
                # Для кириллицы используем разбиение по пробелам и знакам препинания
                stop_playback_commands = ['стоп', 'достаточно', 'хватит', 'понятно']
                is_stop_command = False
                # Разбиваем текст на слова по пробелам и знакам препинания
                import string
                words = re.split(r'[\s\.,!?;:]+', recognized_text_lower)
                for cmd in stop_playback_commands:
                    if cmd in words or recognized_text_lower == cmd:
                        is_stop_command = True
                        logger.info(f"Обнаружена команда остановки: {cmd} в тексте: {phrase.text}")
                        break
                
                # Примечание: проверяем команды завершения разговора
                # Команды: конец разговора, пока, досвидания, до свидания
                end_conversation_commands = ['конец разговора', 'пока', 'досвидания', 'до свидания']
                is_end_command = any(cmd in recognized_text_lower for cmd in end_conversation_commands)
                
                # Отправка распознанного текста
                await ws.send_json({
                    "event": "transcript",
                    "phrase": phrase.to_dict(),
                    "is_stop_command": is_stop_command,
                    "is_end_command": is_end_command
                })
                
                # Примечание: если это команда остановки воспроизведения, не отправляем в LLM
                if is_stop_command:
                    # Устанавливаем флаг остановки для прерывания текущей генерации (если она идет)
                    if not hasattr(ws.app.state, 'stop_flags'):
                        ws.app.state.stop_flags = {}
                    stop_flag_key = id(ws)
                    ws.app.state.stop_flags[stop_flag_key] = True
                    logger.info("Установлен флаг остановки генерации")
                    continue
                
                # Примечание: проверяем, является ли сообщение значимым для отправки в LLM
                if not _is_meaningful_message(phrase.text):
                    logger.info(f"Пропущено незначимое сообщение: '{phrase.text}'")
                    continue
                
                # Формируем контекст из истории сообщений
                context = "\n".join(message_history)
                
                # Примечание: используем потоковую генерацию ответа LLM
                # Накапливаем текст и отправляем по частям, разбивая на абзацы для озвучивания
                full_response = ""
                current_paragraph = ""
                
                # Отправляем событие начала генерации
                await ws.send_json({
                    "event": "llm_response_start"
                })
                
                # Потоковая генерация ответа LLM
                # Примечание: флаг для остановки генерации при получении команды остановки
                # Используем словарь для хранения флагов остановки по ключу соединения
                # Примечание: используем id WebSocket как ключ
                if not hasattr(ws.app.state, 'stop_flags'):
                    ws.app.state.stop_flags = {}
                stop_flag_key = id(ws)
                ws.app.state.stop_flags[stop_flag_key] = False
                
                try:
                    # Примечание: вызываем потоковую генерацию с правильными параметрами
                    for chunk in llm_client.generate_response_stream(
                        phrase.text, 
                        context, 
                        gender, 
                        assistant_name, 
                        system_prompt, 
                        max_tokens
                    ):
                        # Проверяем флаг остановки (может быть установлен из основного цикла)
                        if ws.app.state.stop_flags.get(stop_flag_key, False):
                            logger.info("Прерывание генерации по команде остановки")
                            break
                        
                        if chunk:
                            full_response += chunk
                            current_paragraph += chunk
                            
                            # Отправляем части текста для отображения на клиенте
                            await ws.send_json({
                                "event": "llm_response_chunk",
                                "text": chunk,
                                "full_text": full_response
                            })
                            
                            # Примечание: проверяем, завершился ли абзац (двойной перенос строки или точка + пробел/перенос)
                            # Отправляем абзац на озвучивание если он достаточно длинный и завершен
                            paragraph_complete = False
                            if '\n\n' in current_paragraph:
                                # Двойной перенос строки - явный конец абзаца
                                paragraphs = current_paragraph.split('\n\n', 1)
                                if len(paragraphs) > 1:
                                    paragraph_to_speak = paragraphs[0].strip()
                                    current_paragraph = paragraphs[1]
                                    paragraph_complete = True
                            elif len(current_paragraph) > 100 and (current_paragraph.rstrip().endswith('.') or current_paragraph.rstrip().endswith('!') or current_paragraph.rstrip().endswith('?')):
                                # Абзац достаточно длинный и заканчивается точкой/восклицанием/вопросом
                                # Ищем последнее предложение
                                last_sentence_end = max(
                                    current_paragraph.rfind('. '),
                                    current_paragraph.rfind('! '),
                                    current_paragraph.rfind('? '),
                                    current_paragraph.rfind('.\n'),
                                    current_paragraph.rfind('!\n'),
                                    current_paragraph.rfind('?\n')
                                )
                                if last_sentence_end > 50:  # Минимум 50 символов для озвучивания
                                    paragraph_to_speak = current_paragraph[:last_sentence_end + 1].strip()
                                    current_paragraph = current_paragraph[last_sentence_end + 1:].lstrip()
                                    paragraph_complete = True
                            
                            # Озвучиваем завершенный абзац
                            if paragraph_complete and tts_available:
                                try:
                                    tts_text = _strip_code_for_tts(paragraph_to_speak)
                                    if tts_text and len(tts_text.strip()) > 10:  # Минимум 10 символов для озвучивания
                                        audio_base64 = await tts_client.generate_audio_base64(tts_text, speaker)
                                        if audio_base64:
                                            await ws.send_json({
                                                "event": "tts_audio",
                                                "audio_base64": audio_base64,
                                                "text": paragraph_to_speak
                                            })
                                except Exception as tts_error:
                                    logger.warning(f"Ошибка генерации TTS для абзаца: {tts_error}")
                    
                    # Проверяем, была ли генерация прервана командой остановки
                    was_stopped = ws.app.state.stop_flags.get(stop_flag_key, False)
                    # Сбрасываем флаг остановки после проверки
                    ws.app.state.stop_flags[stop_flag_key] = False
                    
                    # Если генерация была прервана командой остановки, не озвучиваем оставшийся текст
                    if was_stopped:
                        logger.info("Генерация прервана командой остановки, пропускаем озвучивание")
                        # Отправляем событие о прерывании
                        await ws.send_json({
                            "event": "llm_response_interrupted",
                            "text": full_response
                        })
                    else:
                        # Озвучиваем оставшийся текст после завершения потока
                        if current_paragraph.strip() and tts_available:
                            try:
                                tts_text = _strip_code_for_tts(current_paragraph.strip())
                                if tts_text and len(tts_text.strip()) > 10:
                                    audio_base64 = await tts_client.generate_audio_base64(tts_text, speaker)
                                    if audio_base64:
                                        await ws.send_json({
                                            "event": "tts_audio",
                                            "audio_base64": audio_base64,
                                            "text": current_paragraph.strip()
                                        })
                            except Exception as tts_error:
                                logger.warning(f"Ошибка генерации TTS для остатка: {tts_error}")
                    
                    # Отправляем событие завершения генерации (только если не была прервана)
                    if not was_stopped:
                        await ws.send_json({
                            "event": "llm_response_end",
                            "text": full_response
                        })
                    
                    # Добавляем в историю сообщений
                    if full_response:
                        message_history.append(f"Пользователь: {phrase.text}")
                        message_history.append(f"Ассистент: {full_response}")
                        
                        # Ограничиваем историю последними N*2 сообщениями
                        if len(message_history) > MAX_HISTORY * 2:
                            message_history = message_history[-(MAX_HISTORY * 2):]
                            
                except Exception as stream_error:
                    logger.error(f"Ошибка потоковой генерации LLM: {stream_error}")
                    # Fallback: отправляем ошибку
                    await ws.send_json({
                        "event": "llm_response_error",
                        "message": str(stream_error)
                    })
                    
    except WebSocketDisconnect:
        logger.info("WebSocket чата отключен клиентом")
    except Exception as e:
        logger.error(f"Ошибка WebSocket чата: {e}")


@router.websocket("/ws/phone")
async def websocket_phone(ws: WebSocket) -> None:
    """
    WebSocket endpoint для телефонных звонков (Android автоответчик).
    
    Оптимизирован для телефонной линии:
    - Автоматическое приветствие при подключении
    - Низкая задержка обработки
    - Поддержка прерывания речи (barge-in)
    - Таймаут молчания
    
    Протокол:
    1. Клиент подключается, сервер отправляет {"event": "connected"}
    2. Сервер генерирует приветствие и отправляет TTS аудио
    3. Клиент отправляет PCM 16-bit 8kHz mono chunks
    4. Сервер отправляет transcript, llm_response, tts_audio
    5. При получении {"event": "barge_in"} сервер прерывает текущий ответ
    6. При молчании > silence_timeout сервер завершает разговор
    
    Query params:
    - speaker: голос TTS (kseniya, xenia, baya, aidar)
    - caller_id: номер звонящего (опционально, для whitelist)
    """
    await ws.accept()
    
    # Загружаем настройки телефонии из конфига
    phone_config = config.get('phone', {})
    greeting_text = phone_config.get('greeting', 'Здравствуйте! Я голосовой помощник. Чем могу помочь?')
    silence_timeout = phone_config.get('silence_timeout', 30)  # секунд
    whitelist = phone_config.get('whitelist', [])
    blacklist = phone_config.get('blacklist', [])
    auto_answer_enabled = phone_config.get('enabled', True)
    
    # Проверка caller_id по whitelist/blacklist
    caller_id = ws.query_params.get("caller_id", "")
    if blacklist and caller_id in blacklist:
        logger.info(f"Звонок с номера {caller_id} заблокирован (blacklist)")
        await ws.send_json({
            "event": "rejected",
            "reason": "blacklist"
        })
        await ws.close()
        return
    
    if whitelist and caller_id not in whitelist:
        logger.info(f"Звонок с номера {caller_id} заблокирован (не в whitelist)")
        await ws.send_json({
            "event": "rejected",
            "reason": "not_in_whitelist"
        })
        await ws.close()
        return
    
    if not auto_answer_enabled:
        await ws.send_json({
            "event": "rejected",
            "reason": "auto_answer_disabled"
        })
        await ws.close()
        return
    
    if not llm_client.is_available():
        await ws.send_json({
            "event": "error",
            "message": "LLM недоступен"
        })
        await ws.close()
        return
    
    logger.info(f"Телефонный звонок подключен: caller_id={caller_id}")
    
    try:
        pipeline = StreamingASRPipeline()
        context = ""  # Контекст разговора
        tts_client = get_tts_client()
        tts_available = await tts_client.is_ready()
        
        # Настройки голоса
        speaker = ws.query_params.get("speaker", phone_config.get('speaker', 'kseniya'))
        speaker_config_map = {
            "kseniya": {"name": "Маша", "gender": "female"},
            "xenia": {"name": "Женя", "gender": "female"},
            "baya": {"name": "Юля", "gender": "female"},
            "aidar": {"name": "Дима", "gender": "male"},
            "eugene": {"name": "Евгений", "gender": "male"}
        }
        speaker_data = speaker_config_map.get(speaker, {"name": "Маша", "gender": "female"})
        gender = speaker_data["gender"]
        assistant_name = speaker_data["name"]
        
        # Флаг для отслеживания прерывания (barge-in)
        is_speaking = False
        should_interrupt = False
        last_activity_time = asyncio.get_event_loop().time()
        
        # Отправляем событие подключения
        await ws.send_json({
            "event": "connected",
            "caller_id": caller_id,
            "speaker": speaker,
            "assistant_name": assistant_name
        })
        
        # Генерируем и отправляем приветствие
        if tts_available and greeting_text:
            try:
                greeting_audio = await tts_client.generate_audio_base64(greeting_text, speaker)
                if greeting_audio:
                    is_speaking = True
                    await ws.send_json({
                        "event": "greeting",
                        "text": greeting_text,
                        "audio_base64": greeting_audio
                    })
                    # Примечание: клиент должен отправить "greeting_played" когда закончит воспроизведение
            except Exception as e:
                logger.warning(f"Ошибка генерации приветствия: {e}")
                # Отправляем текстовое приветствие без аудио
                await ws.send_json({
                    "event": "greeting",
                    "text": greeting_text
                })
        
        # Обновляем контекст с приветствием
        context = f"Ассистент: {greeting_text}"
        
        # Основной цикл обработки звонка
        async for audio_chunk, is_last in get_chunk_stream(ws, pipeline):
            current_time = asyncio.get_event_loop().time()
            
            # Проверяем таймаут молчания
            if current_time - last_activity_time > silence_timeout:
                logger.info(f"Таймаут молчания ({silence_timeout}с), завершаем звонок")
                await ws.send_json({
                    "event": "timeout",
                    "reason": "silence_timeout"
                })
                break
            
            # Обрабатываем аудио chunk
            phrases = pipeline.process_chunk(audio_chunk, is_last=is_last)
            
            for phrase in phrases:
                last_activity_time = current_time
                
                # Если бот говорит и пользователь заговорил - сигнализируем о прерывании
                if is_speaking:
                    should_interrupt = True
                    await ws.send_json({
                        "event": "barge_in_detected"
                    })
                    is_speaking = False
                
                # Отправка распознанного текста
                await ws.send_json({
                    "event": "transcript",
                    "phrase": phrase.to_dict()
                })
                
                # Примечание: проверяем, является ли сообщение значимым для отправки в LLM
                if not _is_meaningful_message(phrase.text):
                    logger.info(f"Пропущено незначимое сообщение: '{phrase.text}'")
                    continue
                
                # Генерация ответа LLM
                llm_response = llm_client.generate_response(phrase.text, context, gender, assistant_name)
                
                if llm_response:
                    # Отправляем текстовый ответ
                    await ws.send_json({
                        "event": "llm_response",
                        "text": llm_response
                    })
                    
                    # Генерируем и отправляем аудио ответа
                    # Примечание: удаляем блоки кода перед озвучиванием
                    if tts_available:
                        try:
                            tts_text = _strip_code_for_tts(llm_response)
                            audio_base64 = await tts_client.generate_audio_base64(tts_text, speaker)
                            if audio_base64:
                                is_speaking = True
                                await ws.send_json({
                                    "event": "tts_audio",
                                    "audio_base64": audio_base64
                                })
                        except Exception as tts_error:
                            logger.warning(f"Ошибка генерации TTS: {tts_error}")
                    
                    # Обновляем контекст
                    context = f"{context}\nПользователь: {phrase.text}\nАссистент: {llm_response}"
                    # Ограничиваем размер контекста (последние 10 обменов)
                    context_lines = context.split('\n')
                    if len(context_lines) > 20:
                        context = '\n'.join(context_lines[-20:])
        
        # Отправляем событие завершения звонка
        await ws.send_json({
            "event": "call_ended",
            "reason": "normal"
        })
        logger.info(f"Телефонный звонок завершен: caller_id={caller_id}")
                    
    except WebSocketDisconnect:
        logger.info(f"Телефонный звонок отключен клиентом: caller_id={caller_id}")
    except Exception as e:
        logger.error(f"Ошибка телефонного звонка: {e}")
        try:
            await ws.send_json({
                "event": "error",
                "message": str(e)
            })
        except:
            pass


@router.post("/chat")
async def chat_with_llm(request: Request) -> dict:
    """
    HTTP endpoint для текстового чата с LLM.
    
    Body: {"message": "текст сообщения", "context": "опциональный контекст", "speaker": "kseniya", "system_prompt": "опциональный промт", "max_tokens": 2048}
    """
    if not llm_client.is_available():
        raise HTTPException(status_code=503, detail="LLM недоступен")
    
    try:
        data = await request.json()
        message = data.get("message", "")
        context = data.get("context", "")
        speaker = data.get("speaker", "kseniya")
        system_prompt = data.get("system_prompt", "")
        max_tokens = int(data.get("max_tokens", 2048))
        
        # Примечание: маппинг голосов на имена и пол
        speaker_config = {
            "kseniya": {"name": "Маша", "gender": "female"},
            "xenia": {"name": "Женя", "gender": "female"},
            "baya": {"name": "Юля", "gender": "female"},
            "aidar": {"name": "Дима", "gender": "male"},
            "eugene": {"name": "Евгений", "gender": "male"}
        }
        config_data = speaker_config.get(speaker, {"name": "Маша", "gender": "female"})
        gender = config_data["gender"]
        assistant_name = config_data["name"]
        
        if not message:
            raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
        
        # Примечание: проверяем, является ли сообщение значимым для отправки в LLM
        if not _is_meaningful_message(message):
            raise HTTPException(status_code=400, detail="Сообщение слишком короткое или незначимое")
        
        response = llm_client.generate_response(message, context, gender, assistant_name, system_prompt, max_tokens)
        
        if response:
            return {"response": response}
        else:
            raise HTTPException(status_code=500, detail="Не удалось получить ответ от LLM")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON")


@router.post("/chat/stream")
async def chat_with_llm_stream(request: Request):
    """
    HTTP endpoint для потокового чата с LLM.
    
    Body: {"message": "текст сообщения", "context": "опциональный контекст"}
    Returns: Server-Sent Events с частями ответа
    """
    if not llm_client.is_available():
        raise HTTPException(status_code=503, detail="LLM недоступен")
    
    try:
        data = await request.json()
        message = data.get("message", "")
        context = data.get("context", "")
        
        if not message:
            raise HTTPException(status_code=400, detail="Сообщение не может быть пустым")
        
        # Примечание: проверяем, является ли сообщение значимым для отправки в LLM
        if not _is_meaningful_message(message):
            raise HTTPException(status_code=400, detail="Сообщение слишком короткое или незначимое")
        
        async def generate():
            for chunk in llm_client.generate_response_stream(message, context):
                yield f"data: {json.dumps({'text': chunk}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON")


@router.post("/analyze")
async def analyze_text_endpoint(request: Request) -> dict:
    """
    HTTP endpoint для анализа текста через LLM.
    
    Body: {"text": "текст для анализа"}
    """
    if not llm_client.is_available():
        raise HTTPException(status_code=503, detail="LLM недоступен")
    
    try:
        data = await request.json()
        text = data.get("text", "")
        
        if not text:
            raise HTTPException(status_code=400, detail="Текст не может быть пустым")
        
        # Примечание: используем универсальные методы анализа
        analysis = llm_client.analyze_text(text)
        keywords = llm_client.extract_keywords(text)
        summary = llm_client.summarize_text(text)
        
        return {
            "analysis": analysis,
            "keywords": keywords,
            "summary": summary
        }
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON")


@router.get("/llm/test")
async def test_llm_connection() -> dict:
    """Тестирует соединение с LLM сервером."""
    if not llm_client.is_available():
        return {"status": "unavailable", "message": "LLM клиент не инициализирован"}
    
    success = llm_client.test_connection()
    
    return {
        "status": "ok" if success else "error",
        "model": llm_client.model,
        "base_url": llm_client.base_url
    }


@router.get("/llm/models")
async def get_llm_models() -> dict:
    """Получает список доступных моделей из LLM сервера."""
    if not llm_client.is_available():
        return {"models": [], "current": None, "error": "LLM клиент не инициализирован"}
    
    models = llm_client.get_available_models()
    current = llm_client.get_current_model()
    
    return {
        "models": models,
        "current": current
    }


@router.post("/llm/model")
async def set_llm_model(request: Request) -> dict:
    """
    Устанавливает модель для использования.
    
    Body: {"model": "model-id"}
    
    Примечание: модель сохраняется в БД и применяется к текущему клиенту.
    """
    if not llm_client.is_available():
        raise HTTPException(status_code=503, detail="LLM клиент не инициализирован")
    
    try:
        data = await request.json()
        model_id = data.get("model", "")
        
        if not model_id:
            raise HTTPException(status_code=400, detail="Модель не указана")
        
        success = llm_client.set_model(model_id)
        
        if success:
            # Сохраняем выбранную модель в БД для персистентности
            if db_conn:
                set_setting(db_conn, "llm_model", model_id, "Текущая модель LLM")
            
            return {
                "success": True,
                "model": llm_client.get_current_model()
            }
        else:
            raise HTTPException(status_code=500, detail="Не удалось установить модель")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON")


# ==================== Roles Endpoints ====================

@router.get("/roles")
async def get_roles() -> dict:
    """
    Получает список всех ролей помощника.
    """
    if not db_conn:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    
    roles = get_all_roles(db_conn)
    return {"roles": roles}


@router.get("/roles/{role_id}")
async def get_role(role_id: str) -> dict:
    """
    Получает роль по ID.
    """
    if not db_conn:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    
    role = get_role_by_id(db_conn, role_id)
    if not role:
        raise HTTPException(status_code=404, detail="Роль не найдена")
    
    return {"role": role}


@router.post("/roles")
async def add_role(request: Request) -> dict:
    """
    Создает новую роль.
    
    Body: {"name": "Название", "prompt": "Системный промт", "max_tokens": 2048}
    """
    if not db_conn:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    
    try:
        data = await request.json()
        
        if not data.get("name") or not data.get("prompt"):
            raise HTTPException(status_code=400, detail="Название и промт обязательны")
        
        role_id = create_role(db_conn, data)
        
        if role_id:
            return {
                "success": True,
                "id": role_id,
                "message": "Роль создана"
            }
        else:
            raise HTTPException(status_code=500, detail="Не удалось создать роль")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON")


@router.put("/roles/{role_id}")
async def edit_role(role_id: str, request: Request) -> dict:
    """
    Обновляет существующую роль.
    
    Body: {"name": "Название", "prompt": "Системный промт", "max_tokens": 2048}
    """
    if not db_conn:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    
    try:
        data = await request.json()
        
        if not data.get("name") or not data.get("prompt"):
            raise HTTPException(status_code=400, detail="Название и промт обязательны")
        
        success = update_role(db_conn, role_id, data)
        
        if success:
            return {
                "success": True,
                "message": "Роль обновлена"
            }
        else:
            raise HTTPException(status_code=404, detail="Роль не найдена")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON")


@router.delete("/roles/{role_id}")
async def remove_role(role_id: str) -> dict:
    """
    Удаляет роль (только пользовательские, не предустановленные).
    """
    if not db_conn:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    
    success = delete_role(db_conn, role_id)
    
    if success:
        return {
            "success": True,
            "message": "Роль удалена"
        }
    else:
        raise HTTPException(status_code=400, detail="Невозможно удалить роль (возможно, она предустановленная)")


# ==================== Settings Endpoints ====================

@router.get("/settings")
async def get_settings() -> dict:
    """
    Получает все настройки приложения.
    """
    if not db_conn:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    
    settings = get_all_settings(db_conn)
    return {"settings": settings}


@router.get("/settings/llm")
async def get_llm_config() -> dict:
    """
    Получает настройки LLM.
    """
    if not db_conn:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    
    settings = get_llm_settings(db_conn)
    
    # Форматируем для удобного использования на фронтенде
    return {
        "base_url": settings.get("llm_base_url", {}).get("value", ""),
        "model": settings.get("llm_model", {}).get("value", ""),
        "api_key": settings.get("llm_api_key", {}).get("value", ""),
        "max_tokens": int(settings.get("llm_max_tokens", {}).get("value", "2048")),
        "temperature": float(settings.get("llm_temperature", {}).get("value", "0.7")),
        "current_model": llm_client.get_current_model() if llm_client.is_available() else None
    }


@router.put("/settings/llm")
async def update_llm_config(request: Request) -> dict:
    """
    Обновляет настройки LLM.
    
    Body: {"base_url": "...", "model": "...", "api_key": "...", "max_tokens": 2048, "temperature": 0.7}
    """
    if not db_conn:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    
    try:
        data = await request.json()
        
        # Маппинг полей на ключи в БД
        settings_map = {
            "base_url": "llm_base_url",
            "model": "llm_model",
            "api_key": "llm_api_key",
            "max_tokens": "llm_max_tokens",
            "temperature": "llm_temperature"
        }
        
        # Сохраняем в БД
        for field, db_key in settings_map.items():
            if field in data:
                set_setting(db_conn, db_key, str(data[field]))
        
        # Если изменилась модель, применяем к текущему клиенту
        if "model" in data and llm_client.is_available():
            llm_client.set_model(data["model"])
        
        return {
            "success": True,
            "message": "Настройки LLM обновлены",
            "note": "Для применения base_url и api_key требуется перезапуск сервера"
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON")


@router.get("/settings/{key}")
async def get_single_setting(key: str) -> dict:
    """
    Получает значение конкретной настройки.
    """
    if not db_conn:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    
    value = get_setting(db_conn, key)
    if value is None:
        raise HTTPException(status_code=404, detail="Настройка не найдена")
    
    return {"key": key, "value": value}


@router.put("/settings/{key}")
async def update_single_setting(key: str, request: Request) -> dict:
    """
    Обновляет значение конкретной настройки.
    
    Body: {"value": "..."}
    """
    if not db_conn:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    
    try:
        data = await request.json()
        value = data.get("value")
        
        if value is None:
            raise HTTPException(status_code=400, detail="Значение не указано")
        
        success = set_setting(db_conn, key, str(value))
        
        if success:
            return {
                "success": True,
                "key": key,
                "value": value
            }
        else:
            raise HTTPException(status_code=500, detail="Не удалось сохранить настройку")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON")


# ==================== TTS Endpoints ====================

@router.get("/tts/status")
async def tts_status() -> dict:
    """Проверка статуса TTS сервиса."""
    tts_client = get_tts_client()
    is_ready = await tts_client.is_ready()
    status = await tts_client.get_status()
    
    return {
        "ready": is_ready,
        "status": status
    }


@router.get("/tts/speakers")
async def tts_speakers() -> dict:
    """Получение списка доступных голосов TTS."""
    tts_client = get_tts_client()
    speakers = await tts_client.get_available_speakers()
    
    return {
        "speakers": speakers,
        "default": "kseniya"
    }


@router.post("/chat/stop")
async def stop_generation(request: Request) -> dict:
    """
    HTTP endpoint для остановки генерации ответа LLM в текущем WebSocket соединении.
    
    Body: {"websocket_id": "опциональный ID соединения"}
    Returns: {"success": true/false}
    """
    try:
        data = await request.json()
        # Примечание: в реальности нужно использовать session_id или другой идентификатор
        # Для простоты используем заголовок или другой механизм
        # Пока что устанавливаем флаг для всех соединений (в продакшене нужно улучшить)
        
        if not hasattr(request.app.state, 'stop_flags'):
            request.app.state.stop_flags = {}
        
        # Устанавливаем флаг остановки для всех активных соединений
        # Примечание: в продакшене нужно использовать правильный идентификатор соединения
        for key in request.app.state.stop_flags:
            request.app.state.stop_flags[key] = True
        
        logger.info("Получена команда остановки генерации через HTTP endpoint")
        return {"success": True, "message": "Команда остановки отправлена"}
    except Exception as e:
        logger.error(f"Ошибка при обработке команды остановки: {e}")
        return {"success": False, "error": str(e)}


@router.post("/tts/generate")
async def tts_generate(request: Request) -> dict:
    """
    HTTP endpoint для генерации аудио из текста.
    
    Body: {"text": "текст для озвучивания", "speaker": "kseniya", "strip_code": true}
    Returns: {"audio_base64": "...", "success": true}
    
    Примечание: strip_code=true (по умолчанию) удаляет блоки кода перед озвучиванием
    """
    tts_client = get_tts_client()
    
    if not await tts_client.is_ready():
        raise HTTPException(status_code=503, detail="TTS сервис не готов")
    
    try:
        data = await request.json()
        text = data.get("text", "")
        speaker = data.get("speaker", "kseniya")
        strip_code = data.get("strip_code", True)  # По умолчанию удаляем код
        
        if not text:
            raise HTTPException(status_code=400, detail="Текст не может быть пустым")
        
        # Примечание: удаляем блоки кода перед озвучиванием если strip_code=True
        tts_text = _strip_code_for_tts(text) if strip_code else text
        
        audio_base64 = await tts_client.generate_audio_base64(tts_text, speaker)
        
        if audio_base64:
            return {
                "success": True,
                "audio_base64": audio_base64,
                "text_length": len(tts_text),
                "original_length": len(text),
                "speaker": speaker
            }
        else:
            raise HTTPException(status_code=500, detail="Не удалось сгенерировать аудио")
            
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Некорректный JSON")


# ==================== Создание приложения ====================

def create_application() -> FastAPI:
    """Создает и настраивает FastAPI приложение."""
    
    app = FastAPI(
        title="Voice Talker - Голосовой помощник",
        description="Приложение для голосового ввода в реальном времени с обработкой через LLM",
        version=VERSION,
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Настройка CORS
    asr_settings = ASRSettings.from_config(config)
    if asr_settings.cors_allow_all:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Инициализация ASR пайплайна при запуске
    @app.on_event("startup")
    async def startup_event():
        logger.info("Запуск приложения Voice Talker...")
        
        # Инициализация ASR
        success = StreamingASRPipeline.init_pipeline(asr_settings)
        if success:
            logger.info("ASR пайплайн успешно инициализирован")
        else:
            logger.warning("ASR пайплайн не инициализирован - голосовой ввод недоступен")
        
        # Тест LLM соединения
        if llm_client.is_available():
            if llm_client.test_connection():
                logger.info(f"LLM соединение успешно: {llm_client.model} @ {llm_client.base_url}")
            else:
                logger.warning(f"LLM сервер недоступен: {llm_client.base_url}")
        else:
            logger.warning("LLM клиент не инициализирован")
        
        # Инициализация TTS клиента
        tts_url = config.get('tts', {}).get('url', 'http://localhost:5002')
        tts_client = get_tts_client(tts_url)
        tts_ready = await tts_client.is_ready()
        if tts_ready:
            logger.info(f"TTS сервис готов: {tts_url}")
        else:
            logger.warning(f"TTS сервис недоступен: {tts_url} - озвучивание отключено")
    
    # Инициализация клиента текстового API
    app.state.text_api_client = TextAPIClient(asr_settings)
    
    # Подключение роутера API
    app.include_router(router, prefix="/api")
    
    # Монтирование статических файлов
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
        
        # Главная страница
        @app.get("/", response_class=HTMLResponse)
        async def index():
            index_file = static_dir / "index.html"
            if index_file.exists():
                return index_file.read_text(encoding="utf-8")
            return "<h1>Voice Talker</h1><p>Статические файлы не найдены</p>"
    else:
        @app.get("/")
        async def index():
            return {"message": "Voice Talker API", "docs": "/docs"}
    
    return app


# Создание экземпляра приложения
app = create_application()


if __name__ == "__main__":
    import uvicorn
    
    server_config = config.get('server', {})
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 8300)
    
    logger.info(f"Запуск сервера на {host}:{port}")
    uvicorn.run(app, host=host, port=port)
