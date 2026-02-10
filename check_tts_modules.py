#!/usr/bin/env python3
"""
Скрипт проверки модулей для TTS микросервиса
Запускается при старте контейнера для проверки всех зависимостей
"""

import sys
import importlib
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_module(module_name, package_name=None):
    """Проверка наличия модуля"""
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, '__version__', 'unknown')
        logger.info(f"OK {module_name}: {version}")
        return True
    except ImportError as e:
        logger.error(f"FAIL {module_name}: {e}")
        if package_name:
            logger.error(f"   Установите: pip install {package_name}")
        return False

def main():
    """Основная функция проверки"""
    logger.info("Проверка модулей TTS микросервиса...")
    
    # Список критических модулей
    critical_modules = [
        ('torch', 'torch'),
        ('torchaudio', 'torchaudio'),
        ('soundfile', 'soundfile'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('omegaconf', 'omegaconf'),
        ('flask', 'flask'),
        ('flask_cors', 'flask-cors'),
        ('requests', 'requests'),
        ('pydantic', 'pydantic'),
    ]
    
    all_ok = True
    
    for module_name, package_name in critical_modules:
        if not check_module(module_name, package_name):
            all_ok = False
    
    if all_ok:
        logger.info("Все модули установлены корректно")
        
        # Примечание: загрузка модели Silero TTS происходит при запуске сервиса
        # Здесь проверяем только базовые зависимости
        logger.info("Загрузка модели Silero TTS будет выполнена при запуске сервиса")
        
        return 0
    else:
        logger.error("Некоторые модули отсутствуют")
        return 1

if __name__ == "__main__":
    sys.exit(main())
