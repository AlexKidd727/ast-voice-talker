"""
TTS Микросервис для Voice Talker
Отдельный сервис для синтеза речи с использованием Silero TTS
Поддержка моделей v3, v5_ru и v5_cis
"""

import os
import logging
import tempfile
import asyncio
import base64
import re
from typing import Optional, Dict, Any, List
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import threading
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Конфигурация моделей Silero TTS
MODELS_CONFIG = {
    "v3_1_ru": {
        "name": "V3.1 Стандарт",
        "file": "v3_1_ru.pt",
        "url": "https://models.silero.ai/models/tts/ru/v3_1_ru.pt",
        "speakers": ["aidar", "baya", "kseniya", "xenia"],
        "min_size": 50_000_000,  # ~60MB
        "supports_flags": False,  # v3 не поддерживает новые флаги
        "supports_ssml": False
    },
    "v5_ru": {
        "name": "V5 Стандарт",
        "file": "v5_ru.pt",
        "url": "https://models.silero.ai/models/tts/ru/v5_ru.pt",
        "speakers": ["aidar", "baya", "kseniya", "xenia", "eugene"],
        "min_size": 130_000_000,  # ~140MB
        "supports_flags": True,  # v5 поддерживает новые флаги
        "supports_ssml": True
    },
    "v5_cis": {
        "name": "V5 CIS (Новые голоса)",
        "file": "v5_cis_base_nostress.pt",
        "url": "https://models.silero.ai/models/tts/ru/v5_cis_base_nostress.pt",
        "speakers": [
            'ru_aigul', 'ru_albina', 'ru_alexandr', 'ru_alfia', 'ru_alfia2',
            'ru_bogdan', 'ru_dmitriy', 'ru_eduard', 'ru_ekaterina', 'ru_gamat',
            'ru_igor', 'ru_karina', 'ru_kejilgan', 'ru_kermen', 'ru_marat',
            'ru_miyau', 'ru_nurgul', 'ru_oksana', 'ru_onaoy', 'ru_ramilia',
            'ru_roman', 'ru_safarhuja', 'ru_saida', 'ru_sibday', 'ru_vika',
            'ru_zara', 'ru_zhadyra', 'ru_zhazira', 'ru_zinaida'
        ],
        "min_size": 130_000_000,  # ~140MB
        "supports_flags": False,  # v5_cis не поддерживает флаги (nostress версия)
        "supports_ssml": True
    }
}

class TTSService:
    """Сервис для синтеза речи с использованием Silero TTS"""
    
    def __init__(self, model_key: str = None):
        """
        Инициализация TTS сервиса
        
        Args:
            model_key: Ключ модели из MODELS_CONFIG (v3_1_ru, v5_ru, v5_cis)
                      Если None, используется из переменной окружения TTS_MODEL или v5_ru по умолчанию
        """
        self.model = None
        self.speakers = None
        self.sample_rate = 48000
        self.is_initialized = False
        self._initialization_lock = asyncio.Lock()
        self._initialization_thread = None
        
        # Определяем модель для использования
        self.model_key = model_key or os.getenv('TTS_MODEL', 'v5_ru')
        if self.model_key not in MODELS_CONFIG:
            logger.warning(f"Неизвестная модель {self.model_key}, используем v5_ru")
            self.model_key = 'v5_ru'
        
        self.model_config = MODELS_CONFIG[self.model_key]
        logger.info(f"Инициализация TTS сервиса с моделью: {self.model_config['name']} ({self.model_key})")
    
    async def initialize(self) -> bool:
        """Инициализация TTS модели"""
        async with self._initialization_lock:
            if self.is_initialized:
                return True
                
            try:
                logger.info(f"Инициализация TTS сервиса с моделью {self.model_config['name']}...")
                
                # Импортируем torch
                import torch
                import requests
                
                # Получаем конфигурацию модели
                model_file_name = self.model_config['file']
                model_url = self.model_config['url']
                min_size = self.model_config['min_size']
                
                # Пути к локальной модели: 1) Docker volume, 2) папка models рядом со скриптом (чтобы не качать повторно при локальном запуске)
                _script_dir = os.path.dirname(os.path.abspath(__file__))
                _local_models_dir = os.path.join(_script_dir, 'models')
                local_model_path = f'/app/models/{model_file_name}'  # Docker
                local_model_path_win = os.path.join(_local_models_dir, model_file_name)  # локальный запуск (Windows/ Linux)
                
                # Путь к кэшу моделей
                cache_dir = os.path.join(os.environ.get('TORCH_HOME', os.path.expanduser('~/.cache/torch')), 'hub')
                repo_dir = os.path.join(cache_dir, 'snakers4_silero-models_master')
                models_dir = os.path.join(repo_dir, 'models')
                model_file = os.path.join(models_dir, model_file_name)
                
                # Сначала проверяем локальную модель (Docker /app/models или папка models рядом со скриптом)
                model_size_ok = False
                for candidate in (local_model_path, local_model_path_win):
                    if os.path.exists(candidate):
                        file_size = os.path.getsize(candidate)
                        if file_size > min_size:
                            logger.info(f"Модель найдена локально: {candidate} ({file_size // (1024*1024)}MB)")
                            model_file = candidate
                            model_size_ok = True
                            break
                
                # Проверяем наличие модели в кэше (проверяем размер файла)
                if not model_size_ok and os.path.exists(model_file):
                    file_size = os.path.getsize(model_file)
                    if file_size > min_size:
                        logger.info(f"Модель найдена в кэше: {model_file} ({file_size // (1024*1024)}MB)")
                        model_size_ok = True
                    else:
                        logger.warning(f"Модель в кэше повреждена ({file_size // (1024*1024)}MB < {min_size // (1024*1024)}MB), перезагружаем...")
                        try:
                            os.remove(model_file)
                        except:
                            pass
                
                if not model_size_ok:
                    logger.info(f"Модель не найдена, скачиваем {model_file_name}...")
                    
                    # Создаём директорию для кэша
                    os.makedirs(models_dir, exist_ok=True)
                    
                    # Скачиваем модель напрямую с повторными попытками
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            logger.info(f"Скачивание модели из {model_url} (попытка {attempt + 1}/{max_retries})...")
                            response = requests.get(model_url, stream=True, timeout=600)
                            response.raise_for_status()
                            
                            total_size = int(response.headers.get('content-length', 0))
                            downloaded = 0
                            
                            with open(model_file, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                                        downloaded += len(chunk)
                                        if total_size > 0 and downloaded % (10 * 1024 * 1024) == 0:
                                            logger.info(f"Скачано {downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB")
                            
                            # Проверяем размер
                            if os.path.getsize(model_file) > min_size:
                                logger.info(f"Модель скачана: {model_file} ({os.path.getsize(model_file) // (1024*1024)}MB)")
                                break
                            else:
                                logger.warning("Скачанный файл слишком мал, повторяем...")
                                try:
                                    os.remove(model_file)
                                except:
                                    pass
                        except Exception as e:
                            logger.error(f"Ошибка скачивания (попытка {attempt + 1}): {e}")
                            if attempt == max_retries - 1:
                                raise
                
                # Загружаем модель напрямую через torch.package
                logger.info("Загрузка модели Silero TTS...")
                model = torch.package.PackageImporter(model_file).load_pickle("tts_models", "model")
                self.model = model
                
                # Устанавливаем список голосов из конфигурации
                self.speakers = {sp: sp for sp in self.model_config['speakers']}
                self.sample_rate = 48000
                
                # Устанавливаем sample_rate из модели если доступно
                if hasattr(model, 'sample_rate'):
                    self.sample_rate = model.sample_rate
                
                # Модель уже загружена и готова к использованию
                logger.info(f"Модель сохранена: {self.model is not None}, тип: {type(self.model)}")
                
                self.is_initialized = True
                logger.info("TTS сервис инициализирован успешно")
                logger.info(f"Модель: {self.model_config['name']}")
                logger.info(f"Доступные голоса ({len(self.speakers)}): {list(self.speakers.keys())[:10]}{'...' if len(self.speakers) > 10 else ''}")
                logger.info(f"Поддержка флагов: {self.model_config['supports_flags']}")
                logger.info(f"Поддержка SSML: {self.model_config['supports_ssml']}")
                
                return True
                
            except Exception as e:
                logger.error(f"Ошибка инициализации TTS сервиса: {e}")
                import traceback
                logger.error(f"Трассировка: {traceback.format_exc()}")
                return False
    
    def initialize_sync(self) -> bool:
        """Синхронная инициализация TTS модели"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.initialize())
        finally:
            loop.close()
    
    def is_ready(self) -> bool:
        """Проверка готовности TTS сервиса"""
        logger.debug(f"Проверка готовности: initialized={self.is_initialized}, model={self.model is not None}")
        return self.is_initialized and self.model is not None
    
    def _format_text_for_ssml(self, text_chunk: str) -> str:
        """
        Форматирует текст в SSML формат для улучшенного синтеза
        
        Args:
            text_chunk: Текст для форматирования
            
        Returns:
            Текст в формате SSML
        """
        if not text_chunk or not self.model_config['supports_ssml']:
            return text_chunk
        
        # Оборачиваем предложения в <s>...</s> теги
        sentences_with_punc = re.findall(r'[^.!?]+[.!?]', text_chunk, re.UNICODE)
        last_end = 0
        if sentences_with_punc:
            last_match = re.finditer(r'[^.!?]+[.!?]', text_chunk, re.UNICODE)
            for m in last_match:
                last_end = m.end()
            trailing_text = text_chunk[last_end:].strip()
        else:
            trailing_text = text_chunk.strip()
        
        ssml_parts = [f"<s>{s.strip()}</s>" for s in sentences_with_punc]
        if trailing_text:
            ssml_parts.append(f"<s>{trailing_text}</s>")
        
        if not ssml_parts:
            return text_chunk
        
        return f"<speak>{''.join(ssml_parts)}</speak>"
    
    async def generate_audio(
        self, 
        text: str, 
        speaker: str = None,
        put_accent: bool = True,
        put_yo: bool = True,
        put_stress_homo: bool = True,
        put_yo_homo: bool = True,
        use_ssml: bool = True
    ) -> Optional[str]:
        """
        Генерация аудио из текста с разбивкой на части
        
        Args:
            text: Текст для озвучивания
            speaker: Голос для озвучивания (по умолчанию первый доступный)
            put_accent: Автоматическая расстановка ударений (только для v5_ru)
            put_yo: Автоматическая замена е на ё (только для v5_ru)
            put_stress_homo: Ударения у омографов (только для v5_ru)
            put_yo_homo: Буква ё у омографов (только для v5_ru)
            use_ssml: Использовать SSML форматирование (для v5 моделей)
            
        Returns:
            Путь к созданному аудиофайлу или None в случае ошибки
        """
        if not self.is_ready():
            logger.warning("TTS сервис не инициализирован")
            return None
            
        if not text or not text.strip():
            logger.warning("Пустой текст для озвучивания")
            return None
        
        # Определяем голос по умолчанию
        if speaker is None:
            speaker = self.model_config['speakers'][0]
        
        # Проверяем, что голос доступен
        if speaker not in self.speakers:
            logger.warning(f"Голос {speaker} не найден, используем {self.model_config['speakers'][0]}")
            speaker = self.model_config['speakers'][0]
        
        # Проверяем поддержку флагов
        if not self.model_config['supports_flags']:
            # Для моделей без поддержки флагов игнорируем их
            put_accent = False
            put_yo = False
            put_stress_homo = False
            put_yo_homo = False
        
        # Проверяем поддержку SSML
        if not self.model_config['supports_ssml']:
            use_ssml = False
            
        try:
            import soundfile as sf
            import numpy as np
            
            # Очищаем текст от HTML тегов и лишних символов
            clean_text = self._clean_text(text)
            
            # Проверяем, что текст не пустой после обработки
            if not clean_text or not clean_text.strip():
                logger.warning("Текст стал пустым после обработки")
                return None
            
            logger.info(f"Генерация аудио для текста длиной {len(clean_text)} символов")
            logger.info(f"Используемый голос: {speaker}")
            if self.model_config['supports_flags']:
                logger.info(f"Флаги: accent={put_accent}, yo={put_yo}, stress_homo={put_stress_homo}, yo_homo={put_yo_homo}")
            
            # Максимальная длина части для Silero TTS (безопасный лимит)
            MAX_CHUNK_LENGTH = 800
            
            # Разбиваем текст на части если нужно
            if len(clean_text) <= MAX_CHUNK_LENGTH:
                text_chunks = [clean_text]
            else:
                text_chunks = self._split_text_for_tts(clean_text, MAX_CHUNK_LENGTH)
                logger.info(f"Текст разбит на {len(text_chunks)} частей")
            
            # Генерируем аудио для каждой части
            audio_arrays = []
            
            for i, chunk in enumerate(text_chunks, 1):
                if not chunk.strip():
                    continue
                    
                logger.info(f"Генерация части {i}/{len(text_chunks)} ({len(chunk)} символов)")
                
                try:
                    # Форматируем текст в SSML если нужно
                    processed_text = self._format_text_for_ssml(chunk) if use_ssml else chunk
                    
                    # Подготавливаем параметры для apply_tts
                    tts_params = {
                        'speaker': speaker,
                        'sample_rate': self.sample_rate
                    }
                    
                    # Добавляем параметры в зависимости от версии API
                    if self.model_config['supports_ssml'] and use_ssml and '<speak>' in processed_text:
                        tts_params['ssml_text'] = processed_text
                    else:
                        tts_params['text'] = processed_text
                    
                    # Добавляем флаги только для моделей с поддержкой
                    if self.model_config['supports_flags']:
                        tts_params['put_accent'] = put_accent
                        tts_params['put_yo'] = put_yo
                        tts_params['put_stress_homo'] = put_stress_homo
                        tts_params['put_yo_homo'] = put_yo_homo
                    
                    audio = self.model.apply_tts(**tts_params)
                    
                    # Преобразуем в numpy массив
                    if hasattr(audio, 'numpy'):
                        audio_arrays.append(audio.numpy())
                    elif hasattr(audio, 'cpu'):
                        audio_arrays.append(audio.cpu().numpy())
                    else:
                        import torch
                        if isinstance(audio, torch.Tensor):
                            audio_arrays.append(audio.cpu().numpy())
                        else:
                            audio_arrays.append(np.array(audio))
                    
                except Exception as chunk_error:
                    error_msg = str(chunk_error)
                    logger.error(f"Ошибка генерации части {i}: {chunk_error}")
                    
                    # Если часть всё ещё слишком длинная, разбиваем ещё раз
                    if "too long" in error_msg.lower():
                        logger.warning(f"Часть {i} слишком длинная, разбиваем на подчасти")
                        sub_chunks = self._split_text_for_tts(chunk, MAX_CHUNK_LENGTH // 2)
                        
                        for j, sub_chunk in enumerate(sub_chunks, 1):
                            if not sub_chunk.strip():
                                continue
                            try:
                                logger.info(f"  Подчасть {j}/{len(sub_chunks)} ({len(sub_chunk)} символов)")
                                
                                processed_text = self._format_text_for_ssml(sub_chunk) if use_ssml else sub_chunk
                                
                                tts_params = {
                                    'speaker': speaker,
                                    'sample_rate': self.sample_rate
                                }
                                
                                if self.model_config['supports_ssml'] and use_ssml and '<speak>' in processed_text:
                                    tts_params['ssml_text'] = processed_text
                                else:
                                    tts_params['text'] = processed_text
                                
                                if self.model_config['supports_flags']:
                                    tts_params['put_accent'] = put_accent
                                    tts_params['put_yo'] = put_yo
                                    tts_params['put_stress_homo'] = put_stress_homo
                                    tts_params['put_yo_homo'] = put_yo_homo
                                
                                audio = self.model.apply_tts(**tts_params)
                                
                                if hasattr(audio, 'numpy'):
                                    audio_arrays.append(audio.numpy())
                                elif hasattr(audio, 'cpu'):
                                    audio_arrays.append(audio.cpu().numpy())
                                else:
                                    import torch
                                    if isinstance(audio, torch.Tensor):
                                        audio_arrays.append(audio.cpu().numpy())
                                    else:
                                        audio_arrays.append(np.array(audio))
                                        
                            except Exception as sub_error:
                                logger.error(f"  Ошибка подчасти {j}: {sub_error}")
                                # Пропускаем проблемную подчасть
                                continue
                    else:
                        # Пропускаем проблемную часть
                        continue
            
            if not audio_arrays:
                logger.error("Не удалось сгенерировать ни одной части аудио")
                return None
            
            # Объединяем все части аудио
            if len(audio_arrays) == 1:
                combined_audio = audio_arrays[0]
            else:
                # Добавляем небольшую паузу между частями (0.1 сек)
                pause_samples = int(self.sample_rate * 0.1)
                pause = np.zeros(pause_samples)
                
                combined_parts = []
                for i, audio_part in enumerate(audio_arrays):
                    combined_parts.append(audio_part)
                    if i < len(audio_arrays) - 1:  # Не добавляем паузу после последней части
                        combined_parts.append(pause)
                
                combined_audio = np.concatenate(combined_parts)
                logger.info(f"Объединено {len(audio_arrays)} частей аудио")
            
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_path = temp_file.name
            
            # Сохраняем аудио в файл
            sf.write(temp_path, combined_audio, self.sample_rate)
            
            logger.info(f"Аудиофайл создан: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Ошибка генерации аудио: {e}")
            import traceback
            logger.error(f"Трассировка: {traceback.format_exc()}")
            return None
    
    def _split_text_for_tts(self, text: str, max_length: int) -> list:
        """
        Разбивка текста на части для TTS
        
        Args:
            text: Исходный текст
            max_length: Максимальная длина части
            
        Returns:
            Список частей текста
        """
        if len(text) <= max_length:
            return [text]
        
        parts = []
        
        # Сначала пытаемся разбить по предложениям
        # Разделители предложений: . ! ? (с учётом пробела после)
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        current_part = ""
        
        for sentence in sentences:
            # Проверяем, поместится ли предложение
            test_part = current_part + (" " if current_part else "") + sentence
            
            if len(test_part) <= max_length:
                current_part = test_part
            else:
                # Текущая часть готова, сохраняем
                if current_part:
                    parts.append(current_part.strip())
                
                # Если одно предложение слишком длинное, разбиваем по запятым
                if len(sentence) > max_length:
                    sub_parts = self._split_by_commas(sentence, max_length)
                    parts.extend(sub_parts[:-1])
                    current_part = sub_parts[-1] if sub_parts else ""
                else:
                    current_part = sentence
        
        # Добавляем последнюю часть
        if current_part:
            parts.append(current_part.strip())
        
        # Фильтруем пустые части
        parts = [p for p in parts if p.strip()]
        
        return parts if parts else [text[:max_length]]
    
    def _split_by_commas(self, text: str, max_length: int) -> list:
        """
        Разбивка предложения по запятым и другим разделителям
        
        Args:
            text: Исходное предложение
            max_length: Максимальная длина части
            
        Returns:
            Список частей
        """
        if len(text) <= max_length:
            return [text]
        
        parts = []
        
        # Разделители: запятая, точка с запятой, двоеточие, тире
        import re
        segments = re.split(r'(?<=[,;:\-])\s*', text)
        
        current_part = ""
        
        for segment in segments:
            test_part = current_part + segment
            
            if len(test_part) <= max_length:
                current_part = test_part
            else:
                if current_part:
                    parts.append(current_part.strip())
                
                # Если сегмент всё ещё слишком длинный, разбиваем по словам
                if len(segment) > max_length:
                    word_parts = self._split_by_words(segment, max_length)
                    parts.extend(word_parts[:-1])
                    current_part = word_parts[-1] if word_parts else ""
                else:
                    current_part = segment
        
        if current_part:
            parts.append(current_part.strip())
        
        return [p for p in parts if p.strip()] or [text[:max_length]]
    
    def _split_by_words(self, text: str, max_length: int) -> list:
        """
        Разбивка текста по словам
        
        Args:
            text: Исходный текст
            max_length: Максимальная длина части
            
        Returns:
            Список частей
        """
        words = text.split()
        parts = []
        current_part = ""
        
        for word in words:
            test_part = current_part + (" " if current_part else "") + word
            
            if len(test_part) <= max_length:
                current_part = test_part
            else:
                if current_part:
                    parts.append(current_part)
                current_part = word
        
        if current_part:
            parts.append(current_part)
        
        return parts if parts else [text[:max_length]]
    
    def generate_audio_sync(
        self, 
        text: str, 
        speaker: str = None,
        put_accent: bool = True,
        put_yo: bool = True,
        put_stress_homo: bool = True,
        put_yo_homo: bool = True,
        use_ssml: bool = True
    ) -> Optional[str]:
        """Синхронная генерация аудио"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_audio(
                text, speaker, put_accent, put_yo, put_stress_homo, put_yo_homo, use_ssml
            ))
        finally:
            loop.close()
    
    def _convert_table_to_text(self, table_text: str) -> str:
        """
        Преобразует Markdown таблицу в читаемый текст для TTS
        
        Args:
            table_text: Текст таблицы в формате Markdown
            
        Returns:
            Читаемый текст для озвучивания
        """
        lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]
        if len(lines) < 2:
            return table_text
        
        # Первая строка - заголовки
        # Разбиваем по | и убираем пустые элементы в начале/конце
        header_cells = [h.strip() for h in lines[0].split('|')]
        headers = [h for idx, h in enumerate(header_cells) if idx > 0 and idx < len(header_cells) - 1]
        
        if not headers:
            return table_text
        
        # Остальные строки - данные (пропускаем разделитель)
        rows = []
        for line in lines[2:]:
            cells = [c.strip() for c in line.split('|')]
            # Фильтруем: убираем первый и последний элемент (пустые из-за | в начале/конце)
            row_data = [c for idx, c in enumerate(cells) if idx > 0 and idx < len(cells) - 1]
            # Обрезаем до длины заголовков
            rows.append(row_data[:len(headers)])
        
        if not rows:
            return table_text
        
        # Формируем читаемый текст
        text_parts = ['Таблица']
        
        for row_idx, row in enumerate(rows):
            if row_idx > 0:
                text_parts.append('Следующая строка')
            
            for header_idx, header in enumerate(headers):
                if header_idx < len(row):
                    cell_value = row[header_idx] if header_idx < len(row) else ''
                    if cell_value:
                        text_parts.append(f'{header}: {cell_value}')
        
        return '. '.join(text_parts) + '.'
    
    def _clean_text(self, text: str) -> str:
        """
        Очистка и предобработка текста для TTS
        
        Примечание: Для v5_cis можно использовать ударения в формате Silero (+ перед гласной),
        например: "м+олоко", "к+артина". Это улучшит качество произношения.
        """
        import re
        import unicodedata
        
        # 1. Приводим Unicode к нормальной форме
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Обработка ударений (конвертируем символ ударения \u0301 в формат Silero + перед гласной)
        # Это полезно для v5_cis, где нет автоматической расстановки ударений
        text = re.sub(r'([аеёиоуыэюя])\u0301', r'+\1', text)
        
        # 3. Обрабатываем Markdown таблицы перед удалением HTML
        # Формат: | Заголовок | Заголовок |\n| :--- | :--- |\n| Ячейка | Ячейка |
        # Улучшенное регулярное выражение для обработки таблиц с пустыми ячейками
        table_pattern = r'(\|[^\n]*\|\s*\n\|[\s:\-|]+\|\s*\n(?:\|[^\n]*\|\s*\n?)+)'
        def replace_table(match):
            return self._convert_table_to_text(match.group(0))
        text = re.sub(table_pattern, replace_table, text)
        
        # 4. Обрабатываем HTML таблицы
        # Извлекаем содержимое таблиц и преобразуем в текст
        html_table_pattern = r'<table[^>]*>([\s\S]*?)</table>'
        def replace_html_table(match):
            table_content = match.group(1)
            # Извлекаем заголовки из <th> или первой строки <tr>
            headers = re.findall(r'<th[^>]*>([^<]+)</th>', table_content, re.IGNORECASE)
            if not headers:
                # Пытаемся извлечь из первой строки <tr>
                first_row = re.search(r'<tr[^>]*>([\s\S]*?)</tr>', table_content, re.IGNORECASE)
                if first_row:
                    headers = re.findall(r'<td[^>]*>([^<]+)</td>', first_row.group(1), re.IGNORECASE)
            
            # Извлекаем строки данных
            rows = []
            for row_match in re.finditer(r'<tr[^>]*>([\s\S]*?)</tr>', table_content, re.IGNORECASE):
                row_content = row_match.group(1)
                cells = re.findall(r'<td[^>]*>([^<]+)</td>', row_content, re.IGNORECASE)
                if cells:
                    rows.append(cells)
            
            if headers and rows:
                text_parts = ['Таблица']
                for row_idx, row in enumerate(rows):
                    if row_idx > 0:
                        text_parts.append('Следующая строка')
                    for header_idx, header in enumerate(headers):
                        if header_idx < len(row):
                            cell_value = row[header_idx].strip()
                            if cell_value:
                                text_parts.append(f'{header.strip()}: {cell_value}')
                return '. '.join(text_parts) + '.'
            return 'Таблица.'
        
        text = re.sub(html_table_pattern, replace_html_table, text, flags=re.IGNORECASE)
        
        # 5. Убираем оставшиеся HTML теги
        text = re.sub(r'<[^>]*>', '', text)
        
        # 4. Замена специальных символов на слова (но сохраняем + для ударений)
        symbol_replacements = [
            (r'[\[\]{}]', ''),
            (r'[№#]', ' номер '),
            (r'°', ' градус '),
            (r'(?<!\w)\s*\+\s*(?!\w)', ' плюс '),  # + с пробелами = математический плюс
            (r'%', ' процентов '),
            (r'&', ' и '),
            (r'\$', ' долларов '),
            (r'€', ' евро '),
            (r'/', ' дробь '),
            (r'\*', ' умножить на '),
            (r'’', ''),
            (r"'''", ''),  # Примечание: тройные одинарные кавычки, используем двойные кавычки для raw-строки
            (r'…', '...'),
        ]
        
        for pattern, replacement in symbol_replacements:
            text = re.sub(pattern, replacement, text)
        
        # 5. Замена типографики
        character_replacements = {
            '«': '"', '»': '"', '"': '"', '"': '"', '„': '"',
            '—': '-', '–': '-'
        }
        for old_char, new_char in character_replacements.items():
            text = text.replace(old_char, new_char)
        
        # 6. Удаление управляющих символов (категория "C"), но сохраняем + для ударений
        # Проверяем, что + стоит перед гласной (формат ударения Silero)
        text = ''.join(char for char in text if unicodedata.category(char)[0] != "C" or char == '+')
        
        # 7. Убираем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        
        # 8. Убираем множественные знаки препинания
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        return text.strip()
    
    def get_available_speakers(self) -> List[str]:
        """Получение списка доступных голосов"""
        if self.speakers and isinstance(self.speakers, dict):
            return list(self.speakers.keys())
        elif self.speakers and isinstance(self.speakers, str):
            return [self.speakers]
        return []
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о текущей модели"""
        return {
            'model_key': self.model_key,
            'model_name': self.model_config['name'],
            'supports_flags': self.model_config['supports_flags'],
            'supports_ssml': self.model_config['supports_ssml'],
            'speakers': self.get_available_speakers(),
            'sample_rate': self.sample_rate
        }
    
    def cleanup_temp_file(self, file_path: str) -> None:
        """Удаление временного файла"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Временный файл удален: {file_path}")
        except Exception as e:
            logger.warning(f"Ошибка удаления временного файла {file_path}: {e}")

# Глобальный экземпляр сервиса
# Модель выбирается через переменную окружения TTS_MODEL (v3_1_ru, v5_ru, v5_cis)
# По умолчанию используется v5_ru
tts_service = TTSService()

# Инициализация в отдельном потоке
def initialize_tts_background():
    """Инициализация TTS в фоновом режиме"""
    logger.info("Запуск инициализации TTS в фоновом режиме...")
    success = tts_service.initialize_sync()
    if success:
        logger.info("TTS сервис инициализирован в фоновом режиме")
    else:
        logger.error("Ошибка инициализации TTS в фоновом режиме")

# Запускаем инициализацию в фоне
tts_service._initialization_thread = threading.Thread(target=initialize_tts_background, daemon=True)
tts_service._initialization_thread.start()

@app.route('/health', methods=['GET'])
def health_check():
    """Проверка здоровья сервиса"""
    model_info = tts_service.get_model_info() if tts_service.is_ready() else {}
    return jsonify({
        'status': 'healthy',
        'tts_ready': tts_service.is_ready(),
        'model_info': model_info,
        'available_speakers': tts_service.get_available_speakers() if tts_service.is_ready() else []
    })

@app.route('/tts/generate', methods=['POST'])
def generate_tts():
    """Генерация аудио из текста с поддержкой новых параметров v5"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        text = data.get('text', '')
        speaker = data.get('speaker', None)  # По умолчанию первый доступный
        
        # Новые параметры для v5 моделей (опциональные)
        put_accent = data.get('put_accent', True)
        put_yo = data.get('put_yo', True)
        put_stress_homo = data.get('put_stress_homo', True)
        put_yo_homo = data.get('put_yo_homo', True)
        use_ssml = data.get('use_ssml', True)
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        # Проверяем готовность сервиса
        if not tts_service.is_ready():
            return jsonify({
                'error': 'TTS service is not ready',
                'message': 'Service is still initializing'
            }), 503
        
        # Генерируем аудио с новыми параметрами
        audio_path = tts_service.generate_audio_sync(
            text=text,
            speaker=speaker,
            put_accent=put_accent,
            put_yo=put_yo,
            put_stress_homo=put_stress_homo,
            put_yo_homo=put_yo_homo,
            use_ssml=use_ssml
        )
        
        if not audio_path:
            return jsonify({'error': 'Failed to generate audio'}), 500
        
        # Читаем аудиофайл и конвертируем в base64
        try:
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Удаляем временный файл
            tts_service.cleanup_temp_file(audio_path)
            
            model_info = tts_service.get_model_info()
            
            return jsonify({
                'success': True,
                'audio_base64': audio_base64,
                'text_length': len(text),
                'speaker': speaker or model_info.get('speakers', [])[0] if model_info.get('speakers') else 'unknown',
                'model': model_info.get('model_name', 'unknown'),
                'used_flags': {
                    'put_accent': put_accent and model_info.get('supports_flags', False),
                    'put_yo': put_yo and model_info.get('supports_flags', False),
                    'put_stress_homo': put_stress_homo and model_info.get('supports_flags', False),
                    'put_yo_homo': put_yo_homo and model_info.get('supports_flags', False),
                    'use_ssml': use_ssml and model_info.get('supports_ssml', False)
                }
            })
            
        except Exception as e:
            logger.error(f"Ошибка чтения аудиофайла: {e}")
            tts_service.cleanup_temp_file(audio_path)
            return jsonify({'error': 'Failed to read audio file'}), 500
            
    except Exception as e:
        logger.error(f"Ошибка в generate_tts: {e}")
        import traceback
        logger.error(f"Трассировка: {traceback.format_exc()}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/tts/speakers', methods=['GET'])
def get_speakers():
    """Получение списка доступных голосов"""
    speakers = tts_service.get_available_speakers()
    default_speaker = speakers[0] if speakers else None
    return jsonify({
        'speakers': speakers,
        'default': default_speaker,
        'model': tts_service.get_model_info() if tts_service.is_ready() else {}
    })

@app.route('/tts/status', methods=['GET'])
def get_status():
    """Получение статуса сервиса"""
    model_info = tts_service.get_model_info() if tts_service.is_ready() else {}
    return jsonify({
        'ready': tts_service.is_ready(),
        'initialized': tts_service.is_initialized,
        'available_speakers': tts_service.get_available_speakers(),
        'model_info': model_info
    })

@app.route('/tts/models', methods=['GET'])
def get_models():
    """Получение списка доступных моделей"""
    return jsonify({
        'available_models': {k: {
            'name': v['name'],
            'speakers_count': len(v['speakers']),
            'supports_flags': v['supports_flags'],
            'supports_ssml': v['supports_ssml']
        } for k, v in MODELS_CONFIG.items()},
        'current_model': tts_service.get_model_info() if tts_service.is_ready() else None
    })

if __name__ == '__main__':
    # Получаем настройки из переменных окружения
    host = os.getenv('TTS_HOST', '0.0.0.0')
    port = int(os.getenv('TTS_PORT', '5002'))
    debug = os.getenv('TTS_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Запуск TTS микросервиса на {host}:{port}")
    logger.info(f"Режим отладки: {debug}")
    
    app.run(host=host, port=port, debug=debug, threaded=True)
