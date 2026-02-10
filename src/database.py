# Voice Talker - Модуль работы с базой данных
# Исправлено: унифицированы функции, добавлена обработка ошибок

import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def create_connection(db_file):
    """Создает подключение к базе данных SQLite."""
    try:
        # Создаем директорию если не существует
        db_path = Path(db_file)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_file)
        conn.row_factory = sqlite3.Row  # Для доступа к колонкам по имени
        logger.info(f"Подключение к базе данных {db_file} установлено")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Ошибка при подключении к базе данных {db_file}: {e}")
        return None

def create_tables(conn):
    """Создает таблицы, если они не существуют."""
    if not conn:
        logger.error("Нет подключения к базе данных")
        return False
    
    try:
        cursor = conn.cursor()
        
        # Таблица клиентов
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone_number TEXT NOT NULL,
                email TEXT,
                crm_id TEXT,
                needs TEXT,
                whatsapp_sent BOOLEAN DEFAULT 0,
                telegram_sent BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Таблица заявок
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_name TEXT NOT NULL,
                customer_phone TEXT NOT NULL,
                customer_email TEXT,
                needs TEXT,
                status TEXT DEFAULT 'new',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Таблица разговоров
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER,
                audio_file_path TEXT,
                transcription TEXT,
                llm_response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (customer_id) REFERENCES customers (id)
            );
        """)
        
        # Таблица ролей помощника
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS assistant_roles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                max_tokens INTEGER DEFAULT 2048,
                is_default BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Таблица настроек приложения (ключ-значение)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                description TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        conn.commit()
        logger.info("Таблицы базы данных созданы/проверены")
        return True
        
    except sqlite3.Error as e:
        logger.error(f"Ошибка при создании таблиц: {e}")
        return False

def insert_customer(conn, customer_data):
    """Добавляет клиента в базу данных."""
    if not conn:
        logger.error("Нет подключения к базе данных")
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO customers (name, phone_number, email, crm_id, needs) 
            VALUES (?, ?, ?, ?, ?)
        """, (
            customer_data.get('name', ''),
            customer_data.get('phone_number', ''),
            customer_data.get('email', ''),
            customer_data.get('crm_id', ''),
            customer_data.get('needs', '')
        ))
        conn.commit()
        customer_id = cursor.lastrowid
        logger.info(f"Клиент добавлен с ID: {customer_id}")
        return customer_id
    except sqlite3.Error as e:
        logger.error(f"Ошибка при добавлении клиента: {e}")
        return None

def get_customer_by_phone(conn, phone_number):
    """Получает клиента по номеру телефона."""
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers WHERE phone_number = ?", (phone_number,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении клиента по телефону: {e}")
        return None

def get_all_customers(conn):
    """Получает всех клиентов из базы данных."""
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении клиентов: {e}")
        return None

def update_whatsapp_telegram_status(conn, customer_id, whatsapp_sent=False, telegram_sent=False):
    """Обновляет статус отправки презентации через WhatsApp и Telegram."""
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE customers 
            SET whatsapp_sent = ?, telegram_sent = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (whatsapp_sent, telegram_sent, customer_id))
        conn.commit()
        logger.info(f"Статус отправки обновлен для клиента {customer_id}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Ошибка при обновлении статуса отправки: {e}")
        return False

def get_unsent_customers(conn):
    """Получает клиентов, которым еще не была отправлена презентация."""
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM customers 
            WHERE whatsapp_sent = 0 OR telegram_sent = 0
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении клиентов для отправки: {e}")
        return None

def insert_conversation(conn, customer_id, audio_file_path, transcription, llm_response):
    """Добавляет запись о разговоре."""
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO conversations (customer_id, audio_file_path, transcription, llm_response)
            VALUES (?, ?, ?, ?)
        """, (customer_id, audio_file_path, transcription, llm_response))
        conn.commit()
        conversation_id = cursor.lastrowid
        logger.info(f"Разговор записан с ID: {conversation_id}")
        return conversation_id
    except sqlite3.Error as e:
        logger.error(f"Ошибка при записи разговора: {e}")
        return None

def get_conversations_by_customer(conn, customer_id):
    """Получает все разговоры клиента."""
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM conversations 
            WHERE customer_id = ? 
            ORDER BY created_at DESC
        """, (customer_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении разговоров клиента: {e}")
        return None


# ==================== Функции для работы с ролями помощника ====================

# Примечание: предустановленные роли, которые добавляются при первом запуске
DEFAULT_ROLES = [
    {
        "id": "universal",
        "name": "Универсальный помощник",
        "prompt": """Ты универсальный голосовой помощник. Отвечай на любые вопросы пользователя.

Правила:
- Отвечай на русском языке кратко и по существу (2-4 предложения)
- Будь дружелюбным и готовым помочь
- Если не знаешь ответа, честно скажи об этом
- НЕ используй эмодзи и markdown""",
        "max_tokens": 2048,
        "is_default": True
    },
    {
        "id": "programming_teacher",
        "name": "Учитель программирования",
        "prompt": """Ты опытный учитель программирования. Помогаешь изучать программирование.

Правила:
- Объясняй концепции простым языком с примерами кода
- Спроси у пользователя, какой язык программирования его интересует, если не указан
- Давай пошаговые объяснения для сложных тем
- Поощряй вопросы и эксперименты
- Исправляй ошибки мягко и конструктивно
- Давай развернутые ответы с примерами кода
- НЕ используй эмодзи""",
        "max_tokens": 4096,
        "is_default": True
    },
    {
        "id": "fun_companion",
        "name": "Веселый собеседник",
        "prompt": """Ты веселый и позитивный собеседник для поднятия настроения.

Правила:
- Будь энергичным и оптимистичным
- Используй юмор, шутки и забавные истории
- Поддерживай позитивный настрой в разговоре
- Если пользователь грустит, постарайся подбодрить его
- Можешь рассказать анекдот или интересный факт
- Отвечай кратко и весело (2-4 предложения)
- НЕ используй эмодзи""",
        "max_tokens": 1024,
        "is_default": True
    },
    {
        "id": "english_teacher",
        "name": "Учитель английского языка",
        "prompt": """Ты опытный учитель английского языка. Помогаешь изучать английский язык.

Правила:
- Объясняй грамматику и лексику простым языком с примерами
- Исправляй ошибки пользователя мягко, объясняя правильный вариант
- Приводи примеры использования слов и фраз в контексте
- Помогай с произношением, объясняя транскрипцию
- Предлагай полезные фразы и идиомы по теме разговора
- Если пользователь пишет на английском - отвечай на английском с пояснениями на русском
- Поощряй практику и хвали за успехи
- Давай развернутые ответы с примерами
- НЕ используй эмодзи""",
        "max_tokens": 4096,
        "is_default": True
    }
]


def init_default_roles(conn):
    """Инициализирует предустановленные роли, если их нет в базе."""
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        for role in DEFAULT_ROLES:
            # Проверяем, есть ли уже такая роль
            cursor.execute("SELECT id FROM assistant_roles WHERE id = ?", (role["id"],))
            if cursor.fetchone() is None:
                cursor.execute("""
                    INSERT INTO assistant_roles (id, name, prompt, max_tokens, is_default)
                    VALUES (?, ?, ?, ?, ?)
                """, (role["id"], role["name"], role["prompt"], role["max_tokens"], role["is_default"]))
                logger.info(f"Добавлена предустановленная роль: {role['name']}")
        
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Ошибка при инициализации ролей: {e}")
        return False


def get_all_roles(conn):
    """Получает все роли из базы данных."""
    if not conn:
        return []
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, prompt, max_tokens, is_default FROM assistant_roles ORDER BY is_default DESC, name ASC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении ролей: {e}")
        return []


def get_role_by_id(conn, role_id):
    """Получает роль по ID."""
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, prompt, max_tokens, is_default FROM assistant_roles WHERE id = ?", (role_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении роли: {e}")
        return None


def create_role(conn, role_data):
    """Создает новую роль."""
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        role_id = role_data.get("id", f"role_{int(__import__('time').time() * 1000)}")
        
        cursor.execute("""
            INSERT INTO assistant_roles (id, name, prompt, max_tokens, is_default)
            VALUES (?, ?, ?, ?, 0)
        """, (
            role_id,
            role_data.get("name", ""),
            role_data.get("prompt", ""),
            role_data.get("max_tokens", 2048)
        ))
        conn.commit()
        logger.info(f"Роль создана: {role_data.get('name')}")
        return role_id
    except sqlite3.Error as e:
        logger.error(f"Ошибка при создании роли: {e}")
        return None


def update_role(conn, role_id, role_data):
    """Обновляет существующую роль."""
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE assistant_roles 
            SET name = ?, prompt = ?, max_tokens = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (
            role_data.get("name", ""),
            role_data.get("prompt", ""),
            role_data.get("max_tokens", 2048),
            role_id
        ))
        conn.commit()
        logger.info(f"Роль обновлена: {role_id}")
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logger.error(f"Ошибка при обновлении роли: {e}")
        return False


def delete_role(conn, role_id):
    """Удаляет роль (только пользовательские, не предустановленные)."""
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        # Не удаляем предустановленные роли
        cursor.execute("DELETE FROM assistant_roles WHERE id = ? AND is_default = 0", (role_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Роль удалена: {role_id}")
        else:
            logger.warning(f"Роль не удалена (возможно, предустановленная): {role_id}")
        return deleted
    except sqlite3.Error as e:
        logger.error(f"Ошибка при удалении роли: {e}")
        return False


# ==================== Функции для работы с настройками ====================

# Примечание: предустановленные настройки LLM
DEFAULT_SETTINGS = {
    "llm_base_url": {
        "value": "http://192.168.1.250:1234/v1",
        "description": "Базовый URL для LLM API (например, LM Studio)"
    },
    "llm_model": {
        "value": "google/gemma-3-4b",
        "description": "Модель LLM по умолчанию"
    },
    "llm_api_key": {
        "value": "not-needed",
        "description": "API ключ для LLM (для локальных моделей можно оставить 'not-needed')"
    },
    "llm_max_tokens": {
        "value": "2048",
        "description": "Максимальное количество токенов в ответе по умолчанию"
    },
    "llm_temperature": {
        "value": "0.7",
        "description": "Температура генерации (0.0-1.0)"
    }
}


def init_default_settings(conn):
    """Инициализирует настройки по умолчанию, если их нет в базе."""
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        for key, data in DEFAULT_SETTINGS.items():
            # Проверяем, есть ли уже такая настройка
            cursor.execute("SELECT key FROM app_settings WHERE key = ?", (key,))
            if cursor.fetchone() is None:
                cursor.execute("""
                    INSERT INTO app_settings (key, value, description)
                    VALUES (?, ?, ?)
                """, (key, data["value"], data["description"]))
                logger.info(f"Добавлена настройка: {key} = {data['value']}")
        
        conn.commit()
        return True
    except sqlite3.Error as e:
        logger.error(f"Ошибка при инициализации настроек: {e}")
        return False


def get_setting(conn, key):
    """Получает значение настройки по ключу."""
    if not conn:
        return None
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        return row["value"] if row else None
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении настройки {key}: {e}")
        return None


def get_all_settings(conn):
    """Получает все настройки."""
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value, description FROM app_settings ORDER BY key")
        rows = cursor.fetchall()
        return {row["key"]: {"value": row["value"], "description": row["description"]} for row in rows}
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении настроек: {e}")
        return {}


def set_setting(conn, key, value, description=None):
    """Устанавливает значение настройки."""
    if not conn:
        return False
    
    try:
        cursor = conn.cursor()
        
        # Проверяем, существует ли настройка
        cursor.execute("SELECT key FROM app_settings WHERE key = ?", (key,))
        exists = cursor.fetchone() is not None
        
        if exists:
            if description:
                cursor.execute("""
                    UPDATE app_settings 
                    SET value = ?, description = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE key = ?
                """, (value, description, key))
            else:
                cursor.execute("""
                    UPDATE app_settings 
                    SET value = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE key = ?
                """, (value, key))
        else:
            cursor.execute("""
                INSERT INTO app_settings (key, value, description)
                VALUES (?, ?, ?)
            """, (key, value, description or ""))
        
        conn.commit()
        logger.info(f"Настройка обновлена: {key} = {value}")
        return True
    except sqlite3.Error as e:
        logger.error(f"Ошибка при установке настройки {key}: {e}")
        return False


def get_llm_settings(conn):
    """Получает все настройки LLM."""
    if not conn:
        return {}
    
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT key, value, description FROM app_settings WHERE key LIKE 'llm_%' ORDER BY key")
        rows = cursor.fetchall()
        return {row["key"]: {"value": row["value"], "description": row["description"]} for row in rows}
    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении настроек LLM: {e}")
        return {}


def update_llm_settings(conn, settings_dict):
    """Обновляет настройки LLM."""
    if not conn:
        return False
    
    try:
        for key, value in settings_dict.items():
            if key.startswith("llm_"):
                set_setting(conn, key, str(value))
        return True
    except Exception as e:
        logger.error(f"Ошибка при обновлении настроек LLM: {e}")
        return False
