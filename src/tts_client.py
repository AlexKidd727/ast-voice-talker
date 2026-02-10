"""
TTS Клиент для Voice Talker
Клиент для взаимодействия с TTS микросервисом (Silero TTS)
"""

import os
import logging
import base64
import tempfile
import asyncio
import aiohttp
import re
from typing import Optional

from tts_transliteration_data import (
    PYTHON_OPERATOR_SYMBOLS,
    get_english_words_2000,
    get_python_operator_symbols_sorted,
)

logger = logging.getLogger(__name__)

class TTSClient:
    """Клиент для взаимодействия с TTS микросервисом"""
    
    def __init__(self, base_url: str = None):
        """
        Инициализация TTS клиента
        
        Args:
            base_url: URL TTS микросервиса (по умолчанию из env TTS_MICROSERVICE_URL)
        """
        # Примечание: приоритет - переданный параметр, затем env, затем дефолт
        self.base_url = base_url or os.getenv('TTS_MICROSERVICE_URL', 'http://localhost:5002')
        self.timeout = 60  # Таймаут для запросов (увеличен для длинных текстов)
        self._session = None
        logger.info(f"TTS клиент инициализирован: {self.base_url}")
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Получение HTTP сессии"""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self):
        """Закрытие HTTP сессии"""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def is_ready(self) -> bool:
        """Проверка готовности TTS микросервиса"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('tts_ready', False)
                return False
        except Exception as e:
            logger.error(f"Ошибка проверки готовности TTS сервиса: {e}")
            return False
    
    async def get_status(self) -> dict:
        """Получение статуса TTS микросервиса"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/tts/status") as response:
                if response.status == 200:
                    return await response.json()
                return {'ready': False, 'error': f'HTTP {response.status}'}
        except Exception as e:
            logger.error(f"Ошибка получения статуса TTS сервиса: {e}")
            return {'ready': False, 'error': str(e)}
    
    async def get_available_speakers(self) -> list:
        """Получение списка доступных голосов"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/tts/speakers") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('speakers', [])
                return []
        except Exception as e:
            logger.error(f"Ошибка получения списка голосов: {e}")
            return []
    
    async def generate_audio(self, text: str, speaker: str = 'kseniya') -> Optional[bytes]:
        """
        Генерация аудио из текста через TTS микросервис
        
        Args:
            text: Текст для озвучивания
            speaker: Голос для озвучивания (по умолчанию 'kseniya')
            
        Returns:
            Байты аудиофайла (WAV) или None в случае ошибки
        """
        if not text or not text.strip():
            logger.warning("Пустой текст для озвучивания")
            return None
        
        # Предобработка текста для лучшего качества TTS
        processed_text = self._preprocess_text(text)
        
        # Проверяем длину текста и разбиваем на части если нужно
        MAX_CHUNK_LENGTH = 1000  # Максимальная длина части
        
        if len(processed_text) <= MAX_CHUNK_LENGTH:
            # Текст короткий, генерируем как обычно
            return await self._generate_single_audio(processed_text, speaker)
        else:
            # Текст длинный, разбиваем на части и объединяем
            logger.info(f"Текст длиной {len(processed_text)} символов, разбиваем на части")
            return await self._generate_chunked_audio(processed_text, speaker, MAX_CHUNK_LENGTH)
    
    async def generate_audio_base64(self, text: str, speaker: str = 'kseniya') -> Optional[str]:
        """
        Генерация аудио из текста и возврат в формате base64
        
        Args:
            text: Текст для озвучивания
            speaker: Голос для озвучивания
            
        Returns:
            Base64-строка аудио или None в случае ошибки
        """
        audio_bytes = await self.generate_audio(text, speaker)
        if audio_bytes:
            return base64.b64encode(audio_bytes).decode('utf-8')
        return None
    
    async def _generate_single_audio(self, text: str, speaker: str) -> Optional[bytes]:
        """
        Генерация аудио из одного фрагмента текста
        
        Args:
            text: Текст для озвучивания
            speaker: Голос для озвучивания
            
        Returns:
            Байты аудиофайла или None в случае ошибки
        """
        try:
            # Подготавливаем данные для запроса
            payload = {
                'text': text,
                'speaker': speaker
            }
            
            session = await self._get_session()
            async with session.post(
                f"{self.base_url}/tts/generate",
                json=payload
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    if data.get('success') and data.get('audio_base64'):
                        # Декодируем base64 аудио
                        audio_data = base64.b64decode(data['audio_base64'])
                        logger.info(f"Аудио сгенерировано: {len(audio_data)} байт")
                        return audio_data
                    
                    else:
                        logger.error(f"TTS микросервис вернул ошибку: {data.get('error', 'Unknown error')}")
                        return None
                
                elif response.status == 503:
                    logger.warning("TTS микросервис еще не готов")
                    return None
                
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка TTS микросервиса (HTTP {response.status}): {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("Таймаут при обращении к TTS микросервису")
            return None
        except Exception as e:
            logger.error(f"Ошибка при обращении к TTS микросервису: {e}")
            return None
    
    async def _generate_chunked_audio(self, text: str, speaker: str, max_chunk_length: int) -> Optional[bytes]:
        """
        Генерация аудио из текста, разбитого на части
        
        Args:
            text: Исходный текст
            speaker: Голос для озвучивания
            max_chunk_length: Максимальная длина части
            
        Returns:
            Объединенные байты аудио или None в случае ошибки
        """
        try:
            # Разбиваем текст на части
            text_chunks = self._split_long_text(text, max_chunk_length)
            
            if not text_chunks:
                logger.error("Не удалось разбить текст на части")
                return None
            
            logger.info(f"Генерируем аудио для {len(text_chunks)} частей текста")
            
            # Генерируем аудио для каждой части
            audio_parts = []
            for i, chunk in enumerate(text_chunks, 1):
                logger.info(f"Генерируем аудио для части {i}/{len(text_chunks)} ({len(chunk)} символов)")
                
                audio_bytes = await self._generate_single_audio(chunk, speaker)
                if audio_bytes:
                    audio_parts.append(audio_bytes)
                else:
                    logger.error(f"Не удалось сгенерировать аудио для части {i}")
                    return None
            
            if not audio_parts:
                logger.error("Не удалось сгенерировать ни одного аудиофайла")
                return None
            
            # Объединяем аудиофайлы
            logger.info(f"Объединяем {len(audio_parts)} аудиофайлов")
            combined_audio = await self._combine_audio_bytes(audio_parts)
            
            if combined_audio:
                logger.info(f"Объединенное аудио: {len(combined_audio)} байт")
                return combined_audio
            else:
                logger.error("Не удалось объединить аудиофайлы")
                return None
                
        except Exception as e:
            logger.error(f"Ошибка при генерации аудио из частей: {e}")
            return None
    
    async def _combine_audio_bytes(self, audio_parts: list) -> Optional[bytes]:
        """
        Объединение нескольких аудиофайлов (WAV) в один
        
        Args:
            audio_parts: Список байтов аудиофайлов
            
        Returns:
            Объединенные байты аудио или None в случае ошибки
        """
        try:
            import io
            import wave
            
            if len(audio_parts) == 1:
                return audio_parts[0]
            
            # Читаем параметры из первого файла
            first_wav = wave.open(io.BytesIO(audio_parts[0]), 'rb')
            params = first_wav.getparams()
            first_wav.close()
            
            # Объединяем все аудиоданные
            output = io.BytesIO()
            output_wav = wave.open(output, 'wb')
            output_wav.setparams(params)
            
            for audio_bytes in audio_parts:
                wav_file = wave.open(io.BytesIO(audio_bytes), 'rb')
                output_wav.writeframes(wav_file.readframes(wav_file.getnframes()))
                wav_file.close()
            
            output_wav.close()
            return output.getvalue()
            
        except Exception as e:
            logger.error(f"Ошибка при объединении аудиофайлов: {e}")
            return None
    
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
    
    def _preprocess_text(self, text: str) -> str:
        """Предобработка текста для улучшения качества TTS"""
        # Обрабатываем Markdown таблицы перед удалением HTML
        # Улучшенное регулярное выражение для обработки таблиц с пустыми ячейками
        table_pattern = r'(\|[^\n]*\|\s*\n\|[\s:\-|]+\|\s*\n(?:\|[^\n]*\|\s*\n?)+)'
        def replace_table(match):
            return self._convert_table_to_text(match.group(0))
        text = re.sub(table_pattern, replace_table, text)
        
        # Обрабатываем HTML таблицы
        html_table_pattern = r'<table[^>]*>([\s\S]*?)</table>'
        def replace_html_table(match):
            table_content = match.group(1)
            headers = re.findall(r'<th[^>]*>([^<]+)</th>', table_content, re.IGNORECASE)
            if not headers:
                first_row = re.search(r'<tr[^>]*>([\s\S]*?)</tr>', table_content, re.IGNORECASE)
                if first_row:
                    headers = re.findall(r'<td[^>]*>([^<]+)</td>', first_row.group(1), re.IGNORECASE)
            
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
        
        # Убираем HTML теги
        text = re.sub(r'<[^>]*>', '', text)
        
        # Заменяем распространенные сокращения на полные формы
        # Примечание: делаем это до удаления пробелов, чтобы не сломать контекст
        text = self._replace_abbreviations(text)
        
        # Убираем лишние пробелы и переносы строк
        text = re.sub(r'\s+', ' ', text)
        
        # Убираем множественные знаки препинания (но сохраняем одиночные)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        # Транслитерация IT-терминов (имена собственные)
        text = self._transliterate_it_terms(text)
        
        # Транслитерация операторов Python (символы: +=, ==, and, or, def, class и т.д.)
        text = self._transliterate_python_operators(text)
        
        # Дефис озвучивать как "минус" только между цифрами (5-3, 5 - 3). В словах (какой-то) не менять.
        text = re.sub(r'(\d)\s*-\s*(\d)', r'\1 минус \2', text)
        
        # Транслитерация 2000 самых частых английских слов
        text = self._transliterate_english_words(text)
        
        # Конвертируем римские цифры в арабские перед конвертацией в слова
        # Примечание: заменяем только если есть слово "век", иначе заменяем английские буквы на транскрипцию
        text = self._convert_roman_to_arabic(text)
        
        # Заменяем английские буквы на транскрипцию (если нет слова "век" рядом с римскими цифрами)
        text = self._transliterate_english_letters(text)
        
        # Конвертируем числа в слова
        text = self._convert_numbers_to_words(text)
        
        # Заменяем проценты
        text = re.sub(r'(\d+(?:[,\.]\d+)?)\s*%', r'\1 процентов', text)
        
        # Заменяем денежные суммы
        text = re.sub(r'\$(\d+(?:[,\.]\d+)?)(?![A-Za-z])', r'\1 долларов', text)
        text = re.sub(r'(\d+(?:[,\.]\d+)?)\s*рублей', r'\1 рублей', text)
        text = re.sub(r'(\d+(?:[,\.]\d+)?)\s*руб\.?', r'\1 рублей', text)
        
        return text.strip()
    
    def _replace_abbreviations(self, text: str) -> str:
        """
        Заменяет распространенные сокращения на полные формы для лучшего озвучивания
        
        Args:
            text: Исходный текст
            
        Returns:
            Текст с замененными сокращениями
        """
        # Сначала обрабатываем сложные случаи с контекстом (числа + сокращения)
        # "г." после числа - это "года"
        text = re.sub(r'(\d+)\s*г\.', r'\1 года', text)
        
        # "гг." после числа - это "годы"
        text = re.sub(r'(\d+)\s*гг\.', r'\1 годы', text)
        
        # "в." после числа - это "век"
        text = re.sub(r'(\d+)\s*в\.', r'\1 век', text)
        
        # "вв." после числа - это "века"
        text = re.sub(r'(\d+)\s*вв\.', r'\1 века', text)
        
        # Словарь сокращений: сокращение -> полная форма
        # Примечание: порядок важен - более длинные фразы должны идти первыми
        # Используем список кортежей для сохранения порядка
        abbreviations = [
            # Исторические и временные сокращения (длинные первыми)
            ('до н.э.', 'до нашей эры'),
            ('н.э.', 'нашей эры'),
            
            # Общие сокращения (многоточия)
            ('и.т.д.', 'и так далее'),
            ('и.т.п.', 'и тому подобное'),
            ('т.д.', 'так далее'),
            ('т.п.', 'тому подобное'),
            ('т.е.', 'то есть'),
            ('т.к.', 'так как'),
            ('т.о.', 'таким образом'),
            ('т.н.', 'так называемый'),
            
            # Административные сокращения
            ('и.о.', 'исполняющий обязанности'),
            ('и.д.', 'исполняющий должность'),
            
            # Сокращения в документах
            ('подразд.', 'подраздел'),
            ('разд.', 'раздел'),
            ('пп.', 'пункты'),
            ('гл.', 'глава'),
            
            # Временные сокращения (уже обработаны выше, но на всякий случай)
            ('вв.', 'века'),
            ('гг.', 'годы'),
            ('в.', 'век'),
            ('г.', 'года'),
            
            # Административные сокращения
            ('пер.', 'переулок'),
            ('кв.', 'квартира'),
            ('ул.', 'улица'),
            ('пр.', 'проспект'),
            ('пл.', 'площадь'),
            ('стр.', 'страница'),  # страница (не строение, т.к. чаще используется)
            ('д.', 'дом'),
            ('к.', 'корпус'),
            
            # Сокращения в документах
            ('ст.', 'статья'),
            ('стр', 'страница'),  # без точки
            ('т.', 'том'),
            ('ч.', 'часть'),
            ('п.', 'пункт'),
            
            # Научные и академические сокращения
            ('проч.', 'прочее'),
            ('напр.', 'например'),
            ('др.', 'другие'),
            ('ср.', 'сравни'),
            ('см.', 'смотри'),
            
            # Единицы измерения (обрабатываем после контекстных)
            ('кг.', 'килограмм'),
            ('км.', 'километр'),
            ('мм.', 'миллиметр'),
            ('м.', 'метр'),
            ('см.', 'сантиметр'),  # может конфликтовать с "смотри", но реже используется
        ]
        
        # Обрабатываем сокращения
        # Примечание: используем границы слов для точного совпадения, но учитываем точку в конце
        for abbr, full in abbreviations:
            # Экранируем специальные символы
            escaped_abbr = re.escape(abbr)
            # Используем границы слов, но учитываем, что точка может быть в конце
            # Паттерн: начало слова или пробел, затем сокращение, затем конец слова или пробел/знак препинания
            pattern = r'(?<!\w)' + escaped_abbr + r'(?!\w)'
            text = re.sub(pattern, full, text, flags=re.IGNORECASE)
        
        return text
    
    def _transliterate_it_terms(self, text: str) -> str:
        """Транслитерация IT-терминов (компании, технологии, языки программирования)"""
        
        # Словарь IT-терминов: английское слово -> русская транскрипция
        # Примечание: порядок важен - более длинные фразы должны идти первыми
        it_terms = {
            # Компании и бренды
            'OpenAI': 'опенэйай',
            'Google': 'гугл',
            'Microsoft': 'майкрософт',
            'Apple': 'эпл',
            'Amazon': 'амазон',
            'Facebook': 'фейсбук',
            'Meta': 'мета',
            'Netflix': 'нетфликс',
            'Tesla': 'тесла',
            'SpaceX': 'спейсэкс',
            'Twitter': 'твиттер',
            'LinkedIn': 'линкедин',
            'GitHub': 'гитхаб',
            'GitLab': 'гитлаб',
            'Bitbucket': 'битбакет',
            'Stack Overflow': 'стэковерфлоу',
            'StackOverflow': 'стэковерфлоу',
            'JetBrains': 'джетбрейнс',
            'Oracle': 'оракл',
            'IBM': 'айбиэм',
            'Intel': 'интел',
            'AMD': 'эйэмди',
            'NVIDIA': 'энвидиа',
            'Nvidia': 'энвидиа',
            'Huawei': 'хуавей',
            'Samsung': 'самсунг',
            'Xiaomi': 'сяоми',
            'Cisco': 'циско',
            'VMware': 'вивэйр',
            'Salesforce': 'сейлсфорс',
            'Adobe': 'адоби',
            'Slack': 'слак',
            'Zoom': 'зум',
            'Spotify': 'спотифай',
            'Uber': 'убер',
            'Airbnb': 'эйрбиэнби',
            'PayPal': 'пейпал',
            'Stripe': 'страйп',
            'Shopify': 'шопифай',
            'Twilio': 'твилио',
            'Cloudflare': 'клаудфлэйр',
            'DigitalOcean': 'диджиталоушен',
            'Heroku': 'хероку',
            'Vercel': 'версель',
            'Netlify': 'нетлифай',
            'MongoDB': 'монгодиби',
            'Redis': 'редис',
            'Elastic': 'эластик',
            'Elasticsearch': 'эластиксёрч',
            'Splunk': 'сплунк',
            'Datadog': 'датадог',
            'New Relic': 'нью релик',
            'PagerDuty': 'пейджердьюти',
            'Atlassian': 'атлассиан',
            'Jira': 'джира',
            'Confluence': 'конфлюэнс',
            'Trello': 'трелло',
            'Asana': 'асана',
            'Notion': 'ноушен',
            'Figma': 'фигма',
            'Sketch': 'скетч',
            'InVision': 'инвижен',
            'Canva': 'канва',
            'HuggingFace': 'хаггингфейс',
            'Hugging Face': 'хаггинг фейс',
            
            # AI/ML компании и модели
            'Anthropic': 'антропик',
            'Claude': 'клод',
            'ChatGPT': 'чатджипити',
            'GPT': 'джипити',
            'DALL-E': 'далли',
            'Midjourney': 'миджорни',
            'Stable Diffusion': 'стейбл дифьюжен',
            'LLaMA': 'лама',
            'Llama': 'лама',
            'Gemini': 'джемини',
            'Gemma': 'джемма',
            'Mistral': 'мистраль',
            'Cohere': 'кохир',
            'DeepMind': 'дипмайнд',
            'TensorFlow': 'тензорфлоу',
            'PyTorch': 'пайторч',
            'Keras': 'керас',
            'scikit-learn': 'сайкит лёрн',
            'Transformers': 'трансформерс',
            'LangChain': 'лангчейн',
            
            # Языки программирования
            'Python': 'пайтон',
            'JavaScript': 'джаваскрипт',
            'TypeScript': 'тайпскрипт',
            'Java': 'джава',
            'Kotlin': 'котлин',
            'Swift': 'свифт',
            'Rust': 'раст',
            'Golang': 'голанг',
            'Ruby': 'руби',
            'Scala': 'скала',
            'Haskell': 'хаскелл',
            'Clojure': 'кложур',
            'Elixir': 'эликсир',
            'Erlang': 'эрланг',
            'Perl': 'перл',
            'PHP': 'пиэйчпи',
            'C++': 'си плюс плюс',
            'C#': 'си шарп',
            'F#': 'эф шарп',
            'Objective-C': 'обджектив си',
            'Assembly': 'ассемблер',
            'COBOL': 'кобол',
            'Fortran': 'фортран',
            'Pascal': 'паскаль',
            'Delphi': 'дельфи',
            'Lua': 'луа',
            'Groovy': 'груви',
            'Dart': 'дарт',
            'Julia': 'джулия',
            'MATLAB': 'матлаб',
            'SQL': 'эскуэль',
            'NoSQL': 'ноуэскуэль',
            'GraphQL': 'графкуэль',
            'HTML': 'эйчтиэмэль',
            'CSS': 'сиэсэс',
            'SASS': 'сасс',
            'LESS': 'лесс',
            'XML': 'иксэмэль',
            'JSON': 'джейсон',
            'YAML': 'ямл',
            'Markdown': 'маркдаун',
            'LaTeX': 'латех',
            'Bash': 'баш',
            'PowerShell': 'пауэршелл',
            'Vim': 'вим',
            'Emacs': 'имакс',
            
            # Фреймворки и библиотеки
            'React': 'реакт',
            'ReactJS': 'реактджиэс',
            'React Native': 'реакт нейтив',
            'Angular': 'ангуляр',
            'Vue': 'вью',
            'VueJS': 'вьюджиэс',
            'Svelte': 'свелт',
            'Next.js': 'некстджиэс',
            'NextJS': 'некстджиэс',
            'Nuxt': 'накст',
            'Gatsby': 'гэтсби',
            'Express': 'экспресс',
            'NestJS': 'нестджиэс',
            'FastAPI': 'фастэйпиай',
            'Django': 'джанго',
            'Flask': 'фласк',
            'Rails': 'рейлс',
            'Ruby on Rails': 'руби он рейлс',
            'Spring': 'спринг',
            'Spring Boot': 'спринг бут',
            'ASP.NET': 'эйэспи нет',
            'Laravel': 'ларавель',
            'Symfony': 'симфони',
            'Bootstrap': 'бутстрап',
            'Tailwind': 'тейлвинд',
            'Material UI': 'материал юай',
            'Chakra': 'чакра',
            'jQuery': 'джейквери',
            'Lodash': 'лодаш',
            'Axios': 'аксиос',
            'Redux': 'редакс',
            'MobX': 'мобэкс',
            'Webpack': 'вебпак',
            'Vite': 'вит',
            'Rollup': 'роллап',
            'Parcel': 'парсел',
            'Babel': 'бабель',
            'ESLint': 'иэслинт',
            'Prettier': 'преттиер',
            'Jest': 'джест',
            'Mocha': 'мока',
            'Cypress': 'сайпресс',
            'Selenium': 'селениум',
            'Puppeteer': 'паппетир',
            'Playwright': 'плейврайт',
            
            # DevOps и инфраструктура
            'Docker': 'докер',
            'Kubernetes': 'кубернетес',
            'Terraform': 'терраформ',
            'Ansible': 'ансибл',
            'Jenkins': 'дженкинс',
            'GitLab CI': 'гитлаб сиай',
            'GitHub Actions': 'гитхаб экшенс',
            'CircleCI': 'сёрклсиай',
            'Travis CI': 'трэвис сиай',
            'ArgoCD': 'аргосиди',
            'Helm': 'хельм',
            'Prometheus': 'прометеус',
            'Grafana': 'графана',
            'Kibana': 'кибана',
            'Logstash': 'логстэш',
            'Kafka': 'кафка',
            'RabbitMQ': 'рэббитэмкью',
            'Nginx': 'энджинкс',
            'Apache': 'апачи',
            'HAProxy': 'хейпрокси',
            'Consul': 'консул',
            'Vault': 'волт',
            'etcd': 'итсиди',
            
            # Облачные платформы
            'AWS': 'эйдаблъюэс',
            'Azure': 'ажур',
            'GCP': 'джисипи',
            'Google Cloud': 'гугл клауд',
            'Lambda': 'лямбда',
            'EC2': 'иситу',
            'S3': 'эстри',
            'CloudFront': 'клаудфронт',
            'Route 53': 'роут пятьдесят три',
            'RDS': 'ардиэс',
            'DynamoDB': 'динамодиби',
            'SQS': 'эскьюэс',
            'SNS': 'эсэнэс',
            'EKS': 'икс',
            'ECS': 'исиэс',
            'Fargate': 'фаргейт',
            
            # Базы данных
            'PostgreSQL': 'постгрес',
            'Postgres': 'постгрес',
            'MySQL': 'майэскуэль',
            'MariaDB': 'мариадиби',
            'SQLite': 'эскуэлайт',
            'Oracle DB': 'оракл диби',
            'Cassandra': 'кассандра',
            'CouchDB': 'каучдиби',
            'Neo4j': 'неофоджей',
            'InfluxDB': 'инфлаксдиби',
            'TimescaleDB': 'таймскейлдиби',
            'Supabase': 'супабейс',
            'Firebase': 'файрбейс',
            'PlanetScale': 'планетскейл',
            
            # IDE и редакторы
            'VS Code': 'виэс код',
            'VSCode': 'виэс код',
            'Visual Studio': 'вижуал студио',
            'IntelliJ': 'интеллиджей',
            'PyCharm': 'пайчарм',
            'WebStorm': 'вебсторм',
            'PhpStorm': 'пхпсторм',
            'RubyMine': 'рубимайн',
            'GoLand': 'голанд',
            'DataGrip': 'датагрип',
            'Rider': 'райдер',
            'CLion': 'сиилайон',
            'Android Studio': 'андроид студио',
            'Xcode': 'экскод',
            'Sublime Text': 'саблайм текст',
            'Atom': 'атом',
            'Notepad++': 'ноутпад плюс плюс',
            
            # Протоколы и форматы
            'HTTP': 'эйчтитипи',
            'HTTPS': 'эйчтитипиэс',
            'REST': 'рест',
            'RESTful': 'рестфул',
            'gRPC': 'джиарписи',
            'WebSocket': 'вебсокет',
            'OAuth': 'оаус',
            'JWT': 'джейдаблъюти',
            'SSL': 'эсэсэль',
            'TLS': 'тиэлэс',
            'SSH': 'эсэсэйч',
            'FTP': 'эфтипи',
            'SFTP': 'эсэфтипи',
            'TCP': 'тисипи',
            'UDP': 'юдипи',
            'DNS': 'диэнэс',
            'CDN': 'сидиэн',
            'API': 'эйпиай',
            'SDK': 'эсдика',
            'CLI': 'сиэлай',
            'GUI': 'гуи',
            'UI': 'юай',
            'UX': 'юэкс',
            'CI/CD': 'сиай сиди',
            'DevOps': 'девопс',
            'MLOps': 'эмэлопс',
            'SRE': 'эсарии',
            'QA': 'кьюэй',
            'TDD': 'тидиди',
            'BDD': 'бидиди',
            'DDD': 'дидиди',
            'SOLID': 'солид',
            'DRY': 'драй',
            'KISS': 'кисс',
            'YAGNI': 'ягни',
            'OOP': 'оуопи',
            'FP': 'эфпи',
            'MVC': 'эмвиси',
            'MVVM': 'эмвивиэм',
            'MVP': 'эмвипи',
            'SPA': 'спа',
            'SSR': 'эсэсар',
            'SSG': 'эсэсджи',
            'PWA': 'пивиэй',
            'SaaS': 'сас',
            'PaaS': 'пас',
            'IaaS': 'иас',
            'IoT': 'айоти',
            'AI': 'эйай',
            'ML': 'эмэль',
            'NLP': 'энэлпи',
            'CV': 'сиви',
            'LLM': 'элэлэм',
            'RAG': 'раг',
            'RLHF': 'арэлэйчэф',
        }
        
        # Сортируем по длине (более длинные фразы первыми)
        sorted_terms = sorted(it_terms.items(), key=lambda x: len(x[0]), reverse=True)
        
        # Заменяем термины (регистронезависимо, но сохраняем границы слов)
        for eng, rus in sorted_terms:
            # Используем word boundaries для точного совпадения
            pattern = r'\b' + re.escape(eng) + r'\b'
            text = re.sub(pattern, rus, text, flags=re.IGNORECASE)
        
        return text
    
    def _transliterate_python_operators(self, text: str) -> str:
        """
        Транслитерация операторов Python: символы (+, -, ==, +=, ...) и ключевые слова
        (and, or, not, if, def, class, return, True, False, None и т.д.) для озвучивания.
        """
        # Сначала символьные операторы (от длинных к коротким)
        for op in get_python_operator_symbols_sorted():
            text = text.replace(op, PYTHON_OPERATOR_SYMBOLS[op])
        return text
    
    def _transliterate_english_words(self, text: str) -> str:
        """
        Транслитерация 2000 самых частых английских слов в русскую транскрипцию для TTS.
        """
        d = get_english_words_2000()
        # Сортируем по длине (убывание), чтобы длинные совпадения обрабатывались первыми
        for w in sorted(d.keys(), key=len, reverse=True):
            pattern = r'\b' + re.escape(w) + r'\b'
            text = re.sub(pattern, d[w], text, flags=re.IGNORECASE)
        return text
    
    def _convert_roman_to_arabic(self, text: str) -> str:
        """
        Преобразует римские цифры в арабские для последующей конвертации в слова
        Только если рядом есть слово "век" (или "века", "веке", "веком", "веку")
        
        Args:
            text: Исходный текст
            
        Returns:
            Текст с замененными римскими цифрами на арабские (только если есть слово "век")
        """
        # Словарь значений римских цифр
        roman_values = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000,
            'i': 1, 'v': 5, 'x': 10, 'l': 50, 'c': 100, 'd': 500, 'm': 1000
        }
        
        def roman_to_int(roman: str) -> int:
            """
            Конвертирует римское число в арабское
            
            Args:
                roman: Строка с римскими цифрами
                
            Returns:
                Арабское число или None если невалидное
            """
            if not roman:
                return None
            
            # Нормализуем к верхнему регистру для обработки
            roman_upper = roman.upper()
            
            # Проверяем, что все символы - валидные римские цифры
            valid_chars = set('IVXLCDM')
            if not all(c in valid_chars for c in roman_upper):
                return None
            
            # Алгоритм конвертации римских цифр в арабские
            result = 0
            prev_value = 0
            
            # Проходим справа налево
            for char in reversed(roman_upper):
                value = roman_values[char]
                
                # Если текущее значение меньше предыдущего, вычитаем (например, IV = 4)
                if value < prev_value:
                    result -= value
                else:
                    result += value
                
                prev_value = value
            
            # Проверяем валидность результата (должно быть в разумных пределах)
            if result < 1 or result > 3999:
                return None
            
            return result
        
        # Паттерн для поиска римских цифр с проверкой наличия слова "век" рядом
        # Примечание: заменяем только если рядом есть слово "век" (в любой форме)
        def replace_roman_if_vek(match):
            """Заменяет римские цифры только если рядом есть слово 'век'"""
            roman_str = match.group(0)
            # Проверяем контекст: есть ли слово "век" в пределах 30 символов до или после
            start = max(0, match.start() - 30)
            end = min(len(text), match.end() + 30)
            context = text[start:end].lower()
            
            # Проверяем наличие слова "век" в любой форме (век, века, веке, веку, веком, в., вв.)
            if re.search(r'\bвек[аеиому]?\b|\bв\.\b|\bвв\.\b', context):
                arabic = roman_to_int(roman_str)
                if arabic is not None:
                    return str(arabic)
            return roman_str  # Если нет слова "век" или не удалось конвертировать, оставляем как есть
        
        # Ищем римские цифры с границами слов
        # Паттерн: граница слова, затем последовательность римских цифр, затем граница слова
        # Примечание: используем (?<!\w) и (?!\w) для границ слов, чтобы не заменять в середине слова
        pattern = r'(?<!\w)([IVXLCDMivxlcdm]+)(?!\w)'
        text = re.sub(pattern, replace_roman_if_vek, text)
        
        return text
    
    def _transliterate_english_letters(self, text: str) -> str:
        """
        Заменяет английские буквы на транскрипцию для TTS
        Применяется только к английским буквам, которые не были заменены как римские цифры
        (т.е. к тем, рядом с которыми нет слова "век")
        
        Args:
            text: Исходный текст
            
        Returns:
            Текст с замененными английскими буквами на транскрипцию
        """
        # Словарь транскрипции английских букв для TTS
        letter_transcription = {
            'A': 'эй', 'B': 'би', 'C': 'си', 'D': 'ди', 'E': 'и', 'F': 'эф',
            'G': 'джи', 'H': 'эйч', 'I': 'ай', 'J': 'джей', 'K': 'кей', 'L': 'эль',
            'M': 'эм', 'N': 'эн', 'O': 'оу', 'P': 'пи', 'Q': 'кью', 'R': 'ар',
            'S': 'эс', 'T': 'ти', 'U': 'ю', 'V': 'ви', 'W': 'дабл ю', 'X': 'экс',
            'Y': 'уай', 'Z': 'зед',
            'a': 'эй', 'b': 'би', 'c': 'си', 'd': 'ди', 'e': 'и', 'f': 'эф',
            'g': 'джи', 'h': 'эйч', 'i': 'ай', 'j': 'джей', 'k': 'кей', 'l': 'эль',
            'm': 'эм', 'n': 'эн', 'o': 'оу', 'p': 'пи', 'q': 'кью', 'r': 'ар',
            's': 'эс', 't': 'ти', 'u': 'ю', 'v': 'ви', 'w': 'дабл ю', 'x': 'экс',
            'y': 'уай', 'z': 'зед'
        }
        
        # Заменяем одиночные английские буквы на транскрипцию
        # Примечание: заменяем только буквы, которые стоят отдельно (не в словах)
        # и рядом с которыми нет слова "век" (они уже обработаны как римские цифры)
        def replace_letter(match):
            letter = match.group(0)
            # Проверяем контекст: есть ли слово "век" рядом
            start = max(0, match.start() - 20)
            end = min(len(text), match.end() + 20)
            context = text[start:end].lower()
            
            # Если есть слово "век" рядом, не заменяем (это уже обработано как римская цифра)
            if re.search(r'\bвек[аеиому]?\b|\bв\.\b|\bвв\.\b', context):
                return letter
            
            # Заменяем на транскрипцию
            return letter_transcription.get(letter, letter)
        
        # Ищем одиночные английские буквы с границами слов
        # Примечание: заменяем только если это действительно отдельная буква, а не часть слова
        pattern = r'(?<!\w)([A-Za-z])(?!\w)'
        text = re.sub(pattern, replace_letter, text)
        
        return text
    
    def _convert_numbers_to_words(self, text: str) -> str:
        """Конвертация цифр в слова для лучшего озвучивания"""
        
        # Базовые словари
        digit_words = {
            '0': 'ноль', '1': 'один', '2': 'два', '3': 'три', '4': 'четыре',
            '5': 'пять', '6': 'шесть', '7': 'семь', '8': 'восемь', '9': 'девять'
        }
        
        teens = {
            10: 'десять', 11: 'одиннадцать', 12: 'двенадцать', 13: 'тринадцать',
            14: 'четырнадцать', 15: 'пятнадцать', 16: 'шестнадцать', 17: 'семнадцать',
            18: 'восемнадцать', 19: 'девятнадцать'
        }
        
        tens = {
            2: 'двадцать', 3: 'тридцать', 4: 'сорок', 5: 'пятьдесят',
            6: 'шестьдесят', 7: 'семьдесят', 8: 'восемьдесят', 9: 'девяносто'
        }
        
        hundreds = {
            1: 'сто', 2: 'двести', 3: 'триста', 4: 'четыреста',
            5: 'пятьсот', 6: 'шестьсот', 7: 'семьсот', 8: 'восемьсот', 9: 'девятьсот'
        }
        
        def get_declension(num, forms):
            """Получение правильного склонения для числа"""
            if num % 100 in [11, 12, 13, 14]:
                return forms[2]
            elif num % 10 == 1:
                return forms[0]
            elif num % 10 in [2, 3, 4]:
                return forms[1]
            else:
                return forms[2]
        
        def convert_three_digits(num, is_thousands=False):
            """Конвертация трехзначного числа в слова"""
            if num == 0:
                return ''
            
            result = []
            
            # Сотни
            hundreds_digit = num // 100
            if hundreds_digit > 0:
                result.append(hundreds[hundreds_digit])
            
            # Десятки и единицы
            remainder = num % 100
            if 10 <= remainder < 20:
                result.append(teens[remainder])
            elif remainder >= 20:
                tens_digit = remainder // 10
                ones_digit = remainder % 10
                result.append(tens[tens_digit])
                if ones_digit > 0:
                    if is_thousands and ones_digit in [1, 2]:
                        result.append('одна' if ones_digit == 1 else 'две')
                    else:
                        result.append(digit_words[str(ones_digit)])
            elif remainder > 0:
                if is_thousands and remainder in [1, 2]:
                    result.append('одна' if remainder == 1 else 'две')
                else:
                    result.append(digit_words[str(remainder)])
            
            return ' '.join(result)
        
        def number_to_words(num):
            """Конвертация числа в слова"""
            if num == 0:
                return 'ноль'
            
            if num < 0:
                return 'минус ' + number_to_words(-num)
            
            result = []
            
            # Миллиарды
            billions_part = num // 1_000_000_000
            if billions_part > 0:
                billions_text = convert_three_digits(billions_part)
                if billions_text:
                    result.append(billions_text)
                    result.append(get_declension(billions_part, ['миллиард', 'миллиарда', 'миллиардов']))
                num %= 1_000_000_000
            
            # Миллионы
            millions_part = num // 1_000_000
            if millions_part > 0:
                millions_text = convert_three_digits(millions_part)
                if millions_text:
                    result.append(millions_text)
                    result.append(get_declension(millions_part, ['миллион', 'миллиона', 'миллионов']))
                num %= 1_000_000
            
            # Тысячи
            thousands_part = num // 1_000
            if thousands_part > 0:
                thousands_text = convert_three_digits(thousands_part, is_thousands=True)
                if thousands_text:
                    result.append(thousands_text)
                    result.append(get_declension(thousands_part, ['тысяча', 'тысячи', 'тысяч']))
                num %= 1_000
            
            # Единицы
            if num > 0:
                units_text = convert_three_digits(num)
                if units_text:
                    result.append(units_text)
            
            return ' '.join(result) if result else 'ноль'
        
        def replace_number(match):
            num_str = match.group(0)
            try:
                # Обработка дробных чисел
                if ',' in num_str or '.' in num_str:
                    num_str = num_str.replace(',', '.')
                    parts = num_str.split('.')
                    integer_part = int(parts[0])
                    decimal_part = parts[1] if len(parts) > 1 else ''
                    
                    result = number_to_words(integer_part)
                    if decimal_part:
                        result += ' целых '
                        # Озвучиваем дробную часть по цифрам
                        for digit in decimal_part:
                            result += digit_words.get(digit, digit) + ' '
                    return result.strip()
                else:
                    return number_to_words(int(num_str))
            except ValueError:
                return num_str
        
        # Ищем любые числа (включая дробные с точкой или запятой)
        text = re.sub(r'-?\d+(?:[,\.]\d+)?', replace_number, text)
        
        return text
    
    def _split_long_text(self, text: str, max_length: int = 1000) -> list:
        """
        Разбиение длинного текста на части для TTS
        
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
        sentences = text.split('. ')
        current_part = ""
        
        for sentence in sentences:
            sentence_with_punctuation = sentence + '. '
            test_part = current_part + sentence_with_punctuation
            
            if len(test_part) <= max_length:
                current_part = test_part
            else:
                if current_part:
                    parts.append(current_part.rstrip('. '))
                    current_part = sentence_with_punctuation
                else:
                    # Если одно предложение слишком длинное, разбиваем по словам
                    word_parts = self._split_sentence_by_words(sentence, max_length)
                    parts.extend(word_parts[:-1])
                    current_part = word_parts[-1] if word_parts else ""
        
        if current_part:
            parts.append(current_part.rstrip('. '))
        
        # Фильтруем пустые части
        parts = [part for part in parts if part.strip()]
        
        return parts
    
    def _split_sentence_by_words(self, sentence: str, max_length: int) -> list:
        """Разбиение предложения по словам"""
        words = sentence.split()
        parts = []
        current_part = ""
        
        for word in words:
            test_part = current_part + word + ' '
            
            if len(test_part) <= max_length:
                current_part = test_part
            else:
                if current_part:
                    parts.append(current_part.strip())
                    current_part = word + ' '
                else:
                    parts.append(word[:max_length])
        
        if current_part:
            parts.append(current_part.strip())
        
        return parts


# Глобальный экземпляр клиента
_tts_client = None

def get_tts_client(base_url: str = None) -> TTSClient:
    """Получение экземпляра TTS клиента"""
    global _tts_client
    if _tts_client is None:
        _tts_client = TTSClient(base_url)
    return _tts_client

async def initialize_tts_client(base_url: str = None) -> bool:
    """Инициализация TTS клиента"""
    try:
        client = get_tts_client(base_url)
        is_ready = await client.is_ready()
        if is_ready:
            logger.info("TTS клиент инициализирован, микросервис готов")
        else:
            logger.info("TTS клиент инициализирован, микросервис еще не готов")
        return True
    except Exception as e:
        logger.error(f"Ошибка инициализации TTS клиента: {e}")
        return False
