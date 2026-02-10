# Voice Talker - Модуль утилит
# Исправлено: улучшено логирование, добавлены дополнительные функции

import logging
import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

def setup_logging(log_file: str = "app.log", level: int = logging.INFO) -> None:
    """
    Настраивает логирование в файл и консоль.

    Args:
        log_file: Имя файла для записи логов.
        level: Уровень логирования.
    """
    # Создаем директорию для логов если не существует
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Настраиваем форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Настраиваем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Очищаем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Файловый обработчик
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Консольный обработчик
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Логируем начало работы
    logger = logging.getLogger(__name__)
    logger.info(f"Логирование настроено. Файл: {log_file}, Уровень: {level}")

def log_info(message: str) -> None:
    """
    Записывает информационное сообщение в лог.

    Args:
        message: Сообщение для записи в лог.
    """
    logger = logging.getLogger(__name__)
    logger.info(message)

def log_error(message: str) -> None:
    """
    Записывает сообщение об ошибке в лог.

    Args:
        message: Сообщение об ошибке для записи в лог.
    """
    logger = logging.getLogger(__name__)
    logger.error(message)

def log_warning(message: str) -> None:
    """
    Записывает предупреждение в лог.

    Args:
        message: Сообщение-предупреждение для записи в лог.
    """
    logger = logging.getLogger(__name__)
    logger.warning(message)

def log_debug(message: str) -> None:
    """
    Записывает отладочное сообщение в лог.

    Args:
        message: Отладочное сообщение для записи в лог.
    """
    logger = logging.getLogger(__name__)
    logger.debug(message)

def get_environment_variable(var_name: str, default: Any = None) -> Any:
    """
    Получает значение переменной окружения.

    Args:
        var_name: Имя переменной окружения.
        default: Значение по умолчанию, если переменная не найдена.

    Returns:
        Значение переменной окружения или значение по умолчанию.
    """
    value = os.environ.get(var_name)
    return value if value is not None else default

def ensure_directory(path: str) -> bool:
    """
    Создает директорию если она не существует.

    Args:
        path: Путь к директории.

    Returns:
        bool: True если директория создана или уже существует.
    """
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log_error(f"Ошибка при создании директории {path}: {e}")
        return False

def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """
    Сохраняет данные в JSON файл.

    Args:
        data: Данные для сохранения.
        file_path: Путь к файлу.

    Returns:
        bool: True если файл сохранен успешно.
    """
    try:
        # Создаем директорию если не существует
        ensure_directory(Path(file_path).parent)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        log_info(f"Данные сохранены в {file_path}")
        return True
    except Exception as e:
        log_error(f"Ошибка при сохранении JSON в {file_path}: {e}")
        return False

def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Загружает данные из JSON файла.

    Args:
        file_path: Путь к файлу.

    Returns:
        dict: Загруженные данные или None в случае ошибки.
    """
    try:
        if not os.path.exists(file_path):
            log_warning(f"Файл не найден: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        log_info(f"Данные загружены из {file_path}")
        return data
    except Exception as e:
        log_error(f"Ошибка при загрузке JSON из {file_path}: {e}")
        return None

def format_timestamp(timestamp: Optional[datetime] = None) -> str:
    """
    Форматирует временную метку.

    Args:
        timestamp: Временная метка (по умолчанию текущее время).

    Returns:
        str: Отформатированная строка времени.
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime('%Y-%m-%d %H:%M:%S')

def validate_phone_number(phone: str) -> bool:
    """
    Проверяет валидность номера телефона.

    Args:
        phone: Номер телефона для проверки.

    Returns:
        bool: True если номер валиден.
    """
    import re
    
    # Убираем все нецифровые символы кроме +
    clean_phone = re.sub(r'[^\d+]', '', phone)
    
    # Проверяем формат: +7XXXXXXXXXX или 8XXXXXXXXXX
    pattern = r'^(\+7|8)\d{10}$'
    return bool(re.match(pattern, clean_phone))

def validate_email(email: str) -> bool:
    """
    Проверяет валидность email адреса.

    Args:
        email: Email для проверки.

    Returns:
        bool: True если email валиден.
    """
    import re
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def get_file_size_mb(file_path: str) -> float:
    """
    Получает размер файла в мегабайтах.

    Args:
        file_path: Путь к файлу.

    Returns:
        float: Размер файла в МБ.
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        log_error(f"Ошибка при получении размера файла {file_path}: {e}")
        return 0.0

def clean_filename(filename: str) -> str:
    """
    Очищает имя файла от недопустимых символов.

    Args:
        filename: Исходное имя файла.

    Returns:
        str: Очищенное имя файла.
    """
    import re
    
    # Заменяем недопустимые символы на подчеркивания
    clean_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Убираем множественные подчеркивания
    clean_name = re.sub(r'_+', '_', clean_name)
    
    # Убираем подчеркивания в начале и конце
    clean_name = clean_name.strip('_')
    
    return clean_name

if __name__ == '__main__':
    # Тестируем функции
    setup_logging("test.log", logging.DEBUG)
    
    log_info("Тест информационного сообщения")
    log_error("Тест сообщения об ошибке")
    log_warning("Тест предупреждения")
    log_debug("Тест отладочного сообщения")
    
    # Тестируем другие функции
    print(f"Текущее время: {format_timestamp()}")
    print(f"Валидный телефон: {validate_phone_number('+79123456789')}")
    print(f"Валидный email: {validate_email('test@example.com')}")
    print(f"Очищенное имя файла: {clean_filename('test<>file.txt')}")
