# Voice Talker - Модуль для работы с WhatsApp и Telegram
# Исправлено: исправлены конфликты имен, добавлена обработка ошибок

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class WhatsAppTelegramBot:
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация ботов для WhatsApp и Telegram.
        
        Args:
            config: Словарь конфигурации с настройками ботов.
        """
        self.config = config
        
        # Настройки Twilio для WhatsApp
        twilio_config = config.get('twilio', {})
        self.twilio_account_sid = twilio_config.get('account_sid', '')
        self.twilio_auth_token = twilio_config.get('auth_token', '')
        self.twilio_phone_number = twilio_config.get('phone_number', '')
        
        # Настройки Telegram
        telegram_config = config.get('whatsapp_telegram', {})
        self.telegram_bot_token = telegram_config.get('telegram_bot_token', '')
        
        # Инициализация клиентов
        self.twilio_client = None
        self.telegram_bot = None
        
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Инициализирует клиенты для WhatsApp и Telegram."""
        try:
            # Инициализация Twilio для WhatsApp
            if self.twilio_account_sid and self.twilio_auth_token:
                from twilio.rest import Client
                self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)
                logger.info("Twilio клиент инициализирован")
            else:
                logger.warning("Twilio не настроен - WhatsApp недоступен")
            
            # Инициализация Telegram бота
            if self.telegram_bot_token:
                from telegram import Bot
                self.telegram_bot = Bot(token=self.telegram_bot_token)
                logger.info("Telegram бот инициализирован")
            else:
                logger.warning("Telegram не настроен - бот недоступен")
                
        except ImportError as e:
            logger.error(f"Не удалось импортировать необходимые библиотеки: {e}")
        except Exception as e:
            logger.error(f"Ошибка при инициализации клиентов: {e}")
    
    def send_whatsapp_message(self, phone_number: str, message: str) -> bool:
        """
        Отправляет текстовое сообщение через WhatsApp.
        
        Args:
            phone_number: Номер телефона получателя.
            message: Текст сообщения.
            
        Returns:
            bool: True если сообщение отправлено, False иначе.
        """
        if not self.twilio_client:
            logger.error("Twilio клиент не инициализирован")
            return False
        
        try:
            # Форматируем номер телефона для WhatsApp
            if not phone_number.startswith('whatsapp:'):
                phone_number = f"whatsapp:{phone_number}"
            
            if not self.twilio_phone_number.startswith('whatsapp:'):
                from_number = f"whatsapp:{self.twilio_phone_number}"
            else:
                from_number = self.twilio_phone_number
            
            message_obj = self.twilio_client.messages.create(
                body=message,
                from_=from_number,
                to=phone_number
            )
            
            logger.info(f"WhatsApp сообщение отправлено на {phone_number}. SID: {message_obj.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при отправке WhatsApp сообщения на {phone_number}: {e}")
            return False
    
    def send_whatsapp_media(self, phone_number: str, media_url: str, caption: str = "") -> bool:
        """
        Отправляет медиафайл через WhatsApp.
        
        Args:
            phone_number: Номер телефона получателя.
            media_url: URL медиафайла.
            caption: Подпись к файлу.
            
        Returns:
            bool: True если файл отправлен, False иначе.
        """
        if not self.twilio_client:
            logger.error("Twilio клиент не инициализирован")
            return False
        
        try:
            # Форматируем номер телефона для WhatsApp
            if not phone_number.startswith('whatsapp:'):
                phone_number = f"whatsapp:{phone_number}"
            
            if not self.twilio_phone_number.startswith('whatsapp:'):
                from_number = f"whatsapp:{self.twilio_phone_number}"
            else:
                from_number = self.twilio_phone_number
            
            message_obj = self.twilio_client.messages.create(
                body=caption,
                media_url=[media_url],
                from_=from_number,
                to=phone_number
            )
            
            logger.info(f"WhatsApp медиа отправлено на {phone_number}. SID: {message_obj.sid}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при отправке WhatsApp медиа на {phone_number}: {e}")
            return False
    
    def send_telegram_message(self, chat_id: str, message: str) -> bool:
        """
        Отправляет текстовое сообщение в Telegram.
        
        Args:
            chat_id: ID чата получателя.
            message: Текст сообщения.
            
        Returns:
            bool: True если сообщение отправлено, False иначе.
        """
        if not self.telegram_bot:
            logger.error("Telegram бот не инициализирован")
            return False
        
        try:
            self.telegram_bot.send_message(chat_id=chat_id, text=message)
            logger.info(f"Telegram сообщение отправлено в чат {chat_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при отправке Telegram сообщения в чат {chat_id}: {e}")
            return False
    
    def send_telegram_document(self, chat_id: str, file_path: str, caption: str = "") -> bool:
        """
        Отправляет документ в Telegram.
        
        Args:
            chat_id: ID чата получателя.
            file_path: Путь к файлу.
            caption: Подпись к файлу.
            
        Returns:
            bool: True если файл отправлен, False иначе.
        """
        if not self.telegram_bot:
            logger.error("Telegram бот не инициализирован")
            return False
        
        if not os.path.exists(file_path):
            logger.error(f"Файл не найден: {file_path}")
            return False
        
        try:
            with open(file_path, 'rb') as file:
                self.telegram_bot.send_document(
                    chat_id=chat_id,
                    document=file,
                    caption=caption
                )
            logger.info(f"Telegram документ отправлен в чат {chat_id}: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при отправке Telegram документа в чат {chat_id}: {e}")
            return False
    
    def send_presentation_whatsapp(self, phone_number: str, presentation_path: str, message: str = "") -> bool:
        """
        Отправляет презентацию через WhatsApp.
        
        Args:
            phone_number: Номер телефона получателя.
            presentation_path: Путь к файлу презентации или URL.
            message: Сообщение с презентацией.
            
        Returns:
            bool: True если презентация отправлена, False иначе.
        """
        if not message:
            message = "Добро пожаловать! Вот презентация по вашему запросу."
        
        # Если это URL, отправляем как медиа
        if presentation_path.startswith('http'):
            return self.send_whatsapp_media(phone_number, presentation_path, message)
        else:
            # Если это локальный файл, сначала отправляем сообщение
            if self.send_whatsapp_message(phone_number, message):
                logger.info(f"Презентация WhatsApp отправлена на {phone_number}")
                return True
            return False
    
    def send_presentation_telegram(self, chat_id: str, presentation_path: str, message: str = "") -> bool:
        """
        Отправляет презентацию через Telegram.
        
        Args:
            chat_id: ID чата получателя.
            presentation_path: Путь к файлу презентации.
            message: Сообщение с презентацией.
            
        Returns:
            bool: True если презентация отправлена, False иначе.
        """
        if not message:
            message = "Добро пожаловать! Вот презентация по вашему запросу."
        
        # Отправляем документ
        if self.send_telegram_document(chat_id, presentation_path, message):
            logger.info(f"Презентация Telegram отправлена в чат {chat_id}")
            return True
        return False
    
    def send_presentation(self, phone_number: str, chat_id: str, presentation_path: str, message: str = "") -> Dict[str, bool]:
        """
        Отправляет презентацию через оба канала.
        
        Args:
            phone_number: Номер телефона для WhatsApp.
            chat_id: ID чата для Telegram.
            presentation_path: Путь к файлу презентации.
            message: Сообщение с презентацией.
            
        Returns:
            dict: Результаты отправки для каждого канала.
        """
        results = {
            'whatsapp': False,
            'telegram': False
        }
        
        if phone_number:
            results['whatsapp'] = self.send_presentation_whatsapp(phone_number, presentation_path, message)
        
        if chat_id:
            results['telegram'] = self.send_presentation_telegram(chat_id, presentation_path, message)
        
        return results
    
    def is_whatsapp_available(self) -> bool:
        """Проверяет доступность WhatsApp."""
        return self.twilio_client is not None
    
    def is_telegram_available(self) -> bool:
        """Проверяет доступность Telegram."""
        return self.telegram_bot is not None
