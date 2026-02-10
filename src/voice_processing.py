# Voice Talker - Модуль обработки голоса
# Исправлено: улучшена обработка ошибок, добавлена поддержка разных форматов

import speech_recognition as sr
import logging
import os
from pathlib import Path
from typing import Optional, Union
import tempfile

logger = logging.getLogger(__name__)

def transcribe_audio(audio_input: Union[str, bytes], language: str = "ru-RU") -> Optional[str]:
    """
    Преобразует аудиоданные в текст с использованием библиотеки SpeechRecognition.

    Args:
        audio_input: Путь к аудиофайлу (str) или аудиоданные (bytes).
        language: Язык для распознавания (по умолчанию русский).

    Returns:
        str: Текст, полученный из аудиоданных, или None в случае ошибки.
    """
    try:
        recognizer = sr.Recognizer()
        
        # Обработка в зависимости от типа входных данных
        if isinstance(audio_input, str):
            # Это путь к файлу
            audio_file = audio_input
            if not os.path.exists(audio_file):
                logger.error(f"Аудиофайл не найден: {audio_file}")
                return None
        else:
            # Это байты - создаем временный файл
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_input)
                audio_file = temp_file.name
        
        try:
            # Загружаем аудиофайл
            with sr.AudioFile(audio_file) as source:
                logger.info(f"Обработка аудиофайла: {audio_file}")
                # Настраиваем распознаватель для шумной среды
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.record(source)
            
            # Распознаем речь
            logger.info(f"Распознавание речи на языке: {language}")
            text = recognizer.recognize_google(audio, language=language)
            
            logger.info(f"Распознанный текст: {text}")
            return text
            
        except sr.UnknownValueError:
            logger.warning("Не удалось распознать речь - возможно, аудио не содержит речи")
            return None
        except sr.RequestError as e:
            logger.error(f"Ошибка сервиса распознавания речи: {e}")
            return None
        finally:
            # Удаляем временный файл если он был создан
            if isinstance(audio_input, bytes) and os.path.exists(audio_file):
                os.unlink(audio_file)

    except Exception as e:
        logger.error(f"Ошибка при транскрипции аудио: {e}")
        return None

def transcribe_audio_offline(audio_file: str, language: str = "ru") -> Optional[str]:
    """
    Офлайн распознавание речи с использованием Vosk (если установлен).

    Args:
        audio_file: Путь к аудиофайлу.
        language: Язык модели.

    Returns:
        str: Распознанный текст или None.
    """
    try:
        import vosk
        import json
        
        # Проверяем наличие модели Vosk
        model_path = f"vosk-model-{language}"
        if not os.path.exists(model_path):
            logger.warning(f"Модель Vosk {model_path} не найдена, используем онлайн распознавание")
            return transcribe_audio(audio_file, f"{language}-RU")
        
        # Инициализация модели
        model = vosk.Model(model_path)
        rec = vosk.KaldiRecognizer(model, 16000)
        
        # Читаем аудиофайл
        with open(audio_file, 'rb') as f:
            while True:
                data = f.read(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text = result.get('text', '')
                    if text:
                        logger.info(f"Vosk распознал: {text}")
                        return text
        
        # Получаем финальный результат
        result = json.loads(rec.FinalResult())
        text = result.get('text', '')
        if text:
            logger.info(f"Vosk финальный результат: {text}")
            return text
        
        return None
        
    except ImportError:
        logger.warning("Vosk не установлен, используем онлайн распознавание")
        return transcribe_audio(audio_file, f"{language}-RU")
    except Exception as e:
        logger.error(f"Ошибка при офлайн распознавании: {e}")
        return None

def validate_audio_file(audio_file: str) -> bool:
    """
    Проверяет валидность аудиофайла.

    Args:
        audio_file: Путь к аудиофайлу.

    Returns:
        bool: True если файл валиден, False иначе.
    """
    try:
        if not os.path.exists(audio_file):
            logger.error(f"Файл не существует: {audio_file}")
            return False
        
        # Проверяем размер файла (не более 10MB)
        file_size = os.path.getsize(audio_file)
        if file_size > 10 * 1024 * 1024:
            logger.error(f"Файл слишком большой: {file_size} байт")
            return False
        
        # Проверяем расширение
        valid_extensions = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        file_ext = Path(audio_file).suffix.lower()
        if file_ext not in valid_extensions:
            logger.warning(f"Неподдерживаемое расширение файла: {file_ext}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ошибка при проверке аудиофайла: {e}")
        return False

def convert_audio_format(input_file: str, output_file: str = None) -> Optional[str]:
    """
    Конвертирует аудиофайл в формат WAV для лучшего распознавания.

    Args:
        input_file: Путь к входному файлу.
        output_file: Путь к выходному файлу (опционально).

    Returns:
        str: Путь к конвертированному файлу или None.
    """
    try:
        import pydub
        from pydub import AudioSegment
        
        if not output_file:
            output_file = input_file.rsplit('.', 1)[0] + '_converted.wav'
        
        # Загружаем аудио
        audio = AudioSegment.from_file(input_file)
        
        # Конвертируем в моно, 16kHz, 16-bit
        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)
        audio = audio.set_sample_width(2)
        
        # Сохраняем
        audio.export(output_file, format="wav")
        
        logger.info(f"Аудио конвертировано: {input_file} -> {output_file}")
        return output_file
        
    except ImportError:
        logger.warning("pydub не установлен, конвертация недоступна")
        return input_file
    except Exception as e:
        logger.error(f"Ошибка при конвертации аудио: {e}")
        return None
