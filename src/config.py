# Voice Talker - Модуль конфигурации
# Обновлено: добавлена поддержка локального LLM сервера (LM Studio)


import os
import json
from pathlib import Path

def load_config(config_file="data/config.json"):
    """
    Загружает конфигурацию из файла config.json или переменных окружения.
    Приоритет: переменные окружения > config.json
    """
    config = {}
    
    # Путь к файлу конфигурации
    config_path = Path(config_file)
    
    # Загрузка из файла config.json
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Ошибка при загрузке конфигурации из {config_file}: {e}")
            config = {}
    
    # Загрузка из переменных окружения (переопределяют файл)
    # Примечание: добавлены новые переменные для локального LLM
    env_mappings = {
        'OPENAI_API_KEY': ['llm', 'api_key'],
        'LLM_MODEL': ['llm', 'model_name'],
        'LLM_BASE_URL': ['llm', 'base_url'],
        'LLM_MAX_TOKENS': ['llm', 'max_tokens'],
        'LLM_TEMPERATURE': ['llm', 'temperature'],
        'DATABASE_PATH': ['database', 'db_path'],
        'CRM_API_URL': ['crm', 'api_endpoint'],
        'CRM_AUTH_TOKEN': ['crm', 'auth_token'],
        'WHATSAPP_API_KEY': ['whatsapp_telegram', 'whatsapp_api_key'],
        'TELEGRAM_BOT_TOKEN': ['whatsapp_telegram', 'telegram_bot_token'],
        'TWILIO_ACCOUNT_SID': ['twilio', 'account_sid'],
        'TWILIO_AUTH_TOKEN': ['twilio', 'auth_token'],
        'TWILIO_PHONE_NUMBER': ['twilio', 'phone_number'],
        # Настройки ASR (распознавание речи)
        'ASR_MODEL_PATH': ['asr', 'model_path'],
        'ASR_SAMPLE_RATE': ['asr', 'sample_rate'],
        'ASR_ENABLE_TEXT_API': ['asr', 'enable_text_api'],
        'ASR_TEXT_API_URL': ['asr', 'text_api_url'],
        # Настройки сервера
        'SERVER_HOST': ['server', 'host'],
        'SERVER_PORT': ['server', 'port'],
        # Настройки TTS (синтез речи)
        'TTS_MICROSERVICE_URL': ['tts', 'url'],
    }
    
    for env_var, config_path_list in env_mappings.items():
        env_value = os.environ.get(env_var)
        if env_value:
            # Создаем вложенную структуру если нужно
            current = config
            for key in config_path_list[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            # Преобразование типов для числовых значений
            if env_var in ['LLM_MAX_TOKENS', 'ASR_SAMPLE_RATE', 'SERVER_PORT']:
                current[config_path_list[-1]] = int(env_value)
            elif env_var == 'LLM_TEMPERATURE':
                current[config_path_list[-1]] = float(env_value)
            elif env_var == 'ASR_ENABLE_TEXT_API':
                current[config_path_list[-1]] = env_value.lower() in ('true', '1', 'yes')
            else:
                current[config_path_list[-1]] = env_value
    
    # Установка значений по умолчанию
    # Примечание: настройки по умолчанию для локальной модели Gemma
    defaults = {
        'llm': {
            'api_key': 'not-needed',  # Для локальных моделей ключ не требуется
            'model_name': 'google/gemma-3-4b',
            'base_url': 'http://192.168.1.250:1234/v1',  # LM Studio по умолчанию
            'max_tokens': 2048,
            'temperature': 0.7
        },
        'database': {
            'db_path': 'data.db'
        },
        'crm': {
            'api_endpoint': 'http://localhost:8000/api/crm',
            'auth_token': ''
        },
        'whatsapp_telegram': {
            'whatsapp_api_key': '',
            'telegram_bot_token': ''
        },
        'twilio': {
            'account_sid': '',
            'auth_token': '',
            'phone_number': ''
        },
        'voice_processing': {
            'audio_sample_rate': 16000
        },
        'asr': {
            'model_path': None,  # None = загрузка из HuggingFace
            'sample_rate': 8000,  # T-one использует 8kHz
            'enable_text_api': False,
            'text_api_url': 'http://localhost:8586/api/text'
        },
        'server': {
            'host': '0.0.0.0',
            'port': 8300
        },
        'tts': {
            'url': 'http://localhost:5002'  # TTS микросервис (Silero TTS)
        }
    }
    
    # Применяем значения по умолчанию
    for section, values in defaults.items():
        if section not in config:
            config[section] = {}
        for key, value in values.items():
            if key not in config[section]:
                config[section][key] = value
    
    return config

def save_config(config, config_file="data/config.json"):
    """Сохраняет конфигурацию в JSON файл"""
    try:
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Ошибка при сохранении конфигурации в {config_file}: {e}")
        return False

def create_default_config(config_file="data/config.json"):
    """Создает файл конфигурации с настройками по умолчанию"""
    # Примечание: настройки по умолчанию для локальной модели Gemma через LM Studio
    default_config = {
        "llm": {
            "api_key": "not-needed",
            "model_name": "google/gemma-3-4b",
            "base_url": "http://192.168.1.250:1234/v1",
            "max_tokens": 2048,
            "temperature": 0.7
        },
        "database": {
            "db_path": "data.db"
        },
        "crm": {
            "api_endpoint": "http://localhost:8000/api/crm",
            "auth_token": "CRM_AUTH_TOKEN"
        },
        "whatsapp_telegram": {
            "whatsapp_api_key": "YOUR_WHATSAPP_API_KEY",
            "telegram_bot_token": "YOUR_TELEGRAM_BOT_TOKEN"
        },
        "twilio": {
            "account_sid": "YOUR_TWILIO_ACCOUNT_SID",
            "auth_token": "YOUR_TWILIO_AUTH_TOKEN",
            "phone_number": "YOUR_TWILIO_PHONE_NUMBER"
        },
        "voice_processing": {
            "audio_sample_rate": 16000
        },
        "asr": {
            "model_path": None,
            "sample_rate": 8000,
            "enable_text_api": False,
            "text_api_url": "http://localhost:8586/api/text"
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8000
        }
    }
    
    return save_config(default_config, config_file)

if __name__ == "__main__":
    # Создаем конфигурацию по умолчанию если файл не существует
    config_path = "data/config.json"
    if not Path(config_path).exists():
        print(f"Создание файла конфигурации: {config_path}")
        create_default_config(config_path)
    
    # Тестируем загрузку
    config = load_config()
    print("Загруженная конфигурация:")
    print(json.dumps(config, indent=2, ensure_ascii=False))
