# Voice Talker - Веб-бот для голосового общения
# FastAPI приложение с WebSocket поддержкой

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from src.config import load_config
from src.database import create_connection, create_tables, insert_customer, insert_conversation
from src.llm import LLM
from src.voice_processing import transcribe_audio, validate_audio_file
from src.utils import setup_logging, log_info, log_error, log_warning

# Настройка логирования
setup_logging("web_bot.log")
logger = logging.getLogger(__name__)

# Создание FastAPI приложения
app = FastAPI(
    title="Voice Talker Web Bot",
    description="Веб-бот для голосового общения ",
    version="1.0.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Глобальные переменные
config = None
llm = None
db_connection = None
active_connections: Dict[str, WebSocket] = {}

@app.on_event("startup")
async def startup_event():
    """Инициализация при запуске приложения."""
    global config, llm, db_connection
    
    try:
        # Загрузка конфигурации
        config = load_config()
        if not config:
            raise Exception("Не удалось загрузить конфигурацию")
        
        # Инициализация базы данных
        db_path = config.get('database', {}).get('db_path', 'data.db')
        db_connection = create_connection(db_path)
        if not db_connection:
            raise Exception("Не удалось подключиться к базе данных")
        
        create_tables(db_connection)
        log_info("База данных инициализирована")
        
        # Инициализация LLM
        llm = LLM(config)
        if not llm.is_available():
            log_warning("LLM недоступен - некоторые функции могут не работать")
        
        log_info("Веб-бот успешно запущен")
        
    except Exception as e:
        log_error(f"Ошибка при запуске приложения: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Очистка при завершении приложения."""
    global db_connection
    
    if db_connection:
        db_connection.close()
        log_info("Соединение с базой данных закрыто")
    
    log_info("Веб-бот завершен")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Главная страница веб-бота."""
    try:
        html_file = Path(__file__).parent / "templates" / "index.html"
        if html_file.exists():
            return FileResponse(html_file)
        else:
            return HTMLResponse("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Voice Talker Web Bot</title>
                <meta charset="utf-8">
            </head>
            <body>
                <h1>Voice Talker Web Bot</h1>
                <p>Веб-бот для голосового общения </p>
                <p>Используйте WebSocket для подключения к боту</p>
            </body>
            </html>
            """)
    except Exception as e:
        log_error(f"Ошибка при загрузке главной страницы: {e}")
        raise HTTPException(status_code=500, detail="Ошибка загрузки страницы")

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint для голосового общения."""
    await websocket.accept()
    active_connections[client_id] = websocket
    log_info(f"Клиент {client_id} подключился")
    
    try:
        while True:
            # Получаем данные от клиента
            data = await websocket.receive()
            
            if data["type"] == "websocket.receive":
                if "bytes" in data:
                    # Обрабатываем аудиоданные
                    await process_audio_data(websocket, client_id, data["bytes"])
                elif "text" in data:
                    # Обрабатываем текстовые сообщения
                    await process_text_message(websocket, client_id, data["text"])
    
    except WebSocketDisconnect:
        log_info(f"Клиент {client_id} отключился")
    except Exception as e:
        log_error(f"Ошибка в WebSocket соединении с клиентом {client_id}: {e}")
    finally:
        if client_id in active_connections:
            del active_connections[client_id]

async def process_audio_data(websocket: WebSocket, client_id: str, audio_data: bytes):
    """Обрабатывает аудиоданные от клиента."""
    try:
        log_info(f"Получены аудиоданные от клиента {client_id}, размер: {len(audio_data)} байт")
        
        # Транскрипция аудио
        transcription = transcribe_audio(audio_data)
        if not transcription:
            await websocket.send_text("Ошибка: не удалось распознать речь")
            return
        
        # Отправляем транскрипцию клиенту
        await websocket.send_text(f"Распознано: {transcription}")
        
        # Анализ через LLM
        if llm and llm.is_available():
            response = llm.generate_response(transcription)
            if response:
                await websocket.send_text(f"Ответ: {response}")
                
                # Сохраняем разговор в базу данных
                save_conversation(client_id, transcription, response, audio_data)
            else:
                await websocket.send_text("Извините, не удалось сгенерировать ответ")
        else:
            await websocket.send_text("LLM недоступен, используйте текстовый режим")
    
    except Exception as e:
        log_error(f"Ошибка при обработке аудио от клиента {client_id}: {e}")
        await websocket.send_text(f"Ошибка: {str(e)}")

async def process_text_message(websocket: WebSocket, client_id: str, message: str):
    """Обрабатывает текстовые сообщения от клиента."""
    try:
        log_info(f"Получено текстовое сообщение от клиента {client_id}: {message}")
        
        if message.startswith("/"):
            # Обработка команд
            await process_command(websocket, client_id, message)
        else:
            # Обычное сообщение
            if llm and llm.is_available():
                response = llm.generate_response(message)
                if response:
                    await websocket.send_text(f"Ответ: {response}")
                else:
                    await websocket.send_text("Извините, не удалось сгенерировать ответ")
            else:
                await websocket.send_text("LLM недоступен")
    
    except Exception as e:
        log_error(f"Ошибка при обработке текста от клиента {client_id}: {e}")
        await websocket.send_text(f"Ошибка: {str(e)}")

async def process_command(websocket: WebSocket, client_id: str, command: str):
    """Обрабатывает команды от клиента."""
    try:
        if command == "/help":
            help_text = """
            Доступные команды:
            /help - показать эту справку
            /status - статус системы
            /clear - очистить историю
            /info - информация о клиенте
            """
            await websocket.send_text(help_text)
        
        elif command == "/status":
            status = {
                "llm_available": llm.is_available() if llm else False,
                "db_connected": db_connection is not None,
                "active_connections": len(active_connections)
            }
            await websocket.send_text(f"Статус: {status}")
        
        elif command == "/clear":
            await websocket.send_text("История очищена")
        
        elif command == "/info":
            await websocket.send_text(f"ID клиента: {client_id}")
        
        else:
            await websocket.send_text(f"Неизвестная команда: {command}")
    
    except Exception as e:
        log_error(f"Ошибка при обработке команды от клиента {client_id}: {e}")
        await websocket.send_text(f"Ошибка: {str(e)}")

def save_conversation(client_id: str, transcription: str, response: str, audio_data: bytes = None):
    """Сохраняет разговор в базу данных."""
    try:
        if not db_connection:
            return
        
        # Создаем или получаем клиента
        customer_data = {
            "name": f"Клиент_{client_id}",
            "phone_number": f"+{client_id}",
            "email": f"client_{client_id}@example.com",
            "needs": "Веб-чат"
        }
        
        customer_id = insert_customer(db_connection, customer_data)
        if customer_id:
            # Сохраняем разговор
            audio_path = None
            if audio_data:
                # Сохраняем аудиофайл
                audio_dir = Path("web_bot/audio")
                audio_dir.mkdir(exist_ok=True)
                audio_path = audio_dir / f"{client_id}_{len(audio_data)}.wav"
                with open(audio_path, 'wb') as f:
                    f.write(audio_data)
            
            insert_conversation(
                db_connection,
                customer_id,
                str(audio_path) if audio_path else None,
                transcription,
                response
            )
            
            log_info(f"Разговор сохранен для клиента {client_id}")
    
    except Exception as e:
        log_error(f"Ошибка при сохранении разговора: {e}")

@app.post("/upload_audio")
async def upload_audio(
    file: UploadFile = File(...),
    client_id: str = Form(...)
):
    """API endpoint для загрузки аудиофайлов."""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Файл не выбран")
        
        # Проверяем тип файла
        if not file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac', '.ogg')):
            raise HTTPException(status_code=400, detail="Неподдерживаемый формат файла")
        
        # Читаем содержимое файла
        audio_data = await file.read()
        
        # Транскрипция
        transcription = transcribe_audio(audio_data)
        if not transcription:
            raise HTTPException(status_code=400, detail="Не удалось распознать речь")
        
        # Генерация ответа
        response = None
        if llm and llm.is_available():
            response = llm.generate_response(transcription)
        
        # Сохранение разговора
        save_conversation(client_id, transcription, response or "Нет ответа", audio_data)
        
        return {
            "transcription": transcription,
            "response": response,
            "status": "success"
        }
    
    except Exception as e:
        log_error(f"Ошибка при загрузке аудио: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status")
async def get_status():
    """API endpoint для получения статуса системы."""
    return {
        "llm_available": llm.is_available() if llm else False,
        "db_connected": db_connection is not None,
        "active_connections": len(active_connections),
        "config_loaded": config is not None
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
