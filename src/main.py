# Voice Talker - Основной модуль приложения
# Исправлено: убрано дублирование кода, исправлены импорты

import logging
import os
import sys
from pathlib import Path

# Добавляем корневую папку проекта в путь
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.database import create_connection, create_tables, insert_customer
from src.llm import LLM
from src.crm import create_ticket
from src.whatsapp_telegram_bot import WhatsAppTelegramBot
from src.voice_processing import transcribe_audio
from src.utils import setup_logging, log_info, log_error

def main():
    """Основная функция приложения"""
    try:
        # Настройка логирования
        setup_logging("app.log")
        log_info("Запуск приложения Voice Talker")
        
        # Загрузка конфигурации
        config = load_config()
        if not config:
            log_error("Не удалось загрузить конфигурацию")
            return
        
        # Инициализация базы данных
        db_path = config.get('database', {}).get('db_path', 'data.db')
        conn = create_connection(db_path)
        if not conn:
            log_error("Не удалось подключиться к базе данных")
            return
        
        create_tables(conn)
        log_info("База данных инициализирована")
        
        # Инициализация LLM
        llm = LLM(config)
        
        # Инициализация ботов
        bot = WhatsAppTelegramBot(config)
        
        # Основной цикл приложения
        log_info("Приложение готово к работе. Введите 'exit' для выхода.")
        
        while True:
            try:
                audio_file = input("\nВведите путь к аудиофайлу (или 'exit'): ").strip()
                
                if audio_file.lower() == 'exit':
                    break
                
                if not os.path.exists(audio_file):
                    print(f"Файл {audio_file} не найден")
                    continue
                
                # Обработка аудио
                log_info(f"Обработка файла: {audio_file}")
                transcription = transcribe_audio(audio_file)
                
                if not transcription:
                    log_error("Не удалось распознать аудио")
                    continue
                
                print(f"Распознанный текст: {transcription}")
                
                # Анализ потребностей через LLM
                needs = llm.analyze_voice(transcription)
                if not needs:
                    log_error("Не удалось проанализировать потребности")
                    continue
                
                print(f"Выявленные потребности: {needs}")
                
                # Создание заявки в CRM
                customer_data = {
                    "name": "Клиент",
                    "phone": "+1234567890",
                    "email": "client@example.com"
                }
                
                ticket_id = create_ticket(customer_data, [needs])
                if ticket_id:
                    log_info(f"Заявка создана с ID: {ticket_id}")
                    
                    # Сохранение в базу данных
                    customer_data['needs'] = needs
                    customer_data['crm_id'] = str(ticket_id)
                    customer_id = insert_customer(conn, customer_data)
                    
                    if customer_id:
                        log_info(f"Клиент сохранен в БД с ID: {customer_id}")
                        
                        # Отправка презентации (заглушка)
                        print("Презентация будет отправлена клиенту")
                    else:
                        log_error("Не удалось сохранить клиента в БД")
                else:
                    log_error("Не удалось создать заявку в CRM")
                    
            except KeyboardInterrupt:
                print("\nПрерывание пользователем")
                break
            except Exception as e:
                log_error(f"Ошибка в основном цикле: {e}")
                print(f"Ошибка: {e}")
        
    except Exception as e:
        log_error(f"Критическая ошибка: {e}")
        print(f"Критическая ошибка: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
        log_info("Приложение завершено")

if __name__ == "__main__":
    main()
