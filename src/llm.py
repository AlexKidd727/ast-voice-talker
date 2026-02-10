# Voice Talker - Модуль работы с LLM
# Обновлено: использование OpenAI 2.x SDK с поддержкой локальной модели Gemma


from openai import OpenAI
import logging
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class LLM:
    """
    Класс для работы с LLM через OpenAI-совместимый API.
    Поддерживает как OpenAI API, так и локальные модели (LM Studio, Ollama и др.)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Инициализация LLM клиента.
        
        Args:
            config: Словарь конфигурации с настройками LLM
        """
        self.config = config
        
        # Получение настроек LLM из конфигурации
        llm_config = config.get('llm', {})
        
        # API ключ (для локальных моделей может быть пустым или любым значением)
        self.api_key = llm_config.get('api_key', 'not-needed')
        
        # Базовый URL для API (для локальных моделей)
        self.base_url = llm_config.get('base_url', 'http://192.168.1.250:1234/v1')
        
        # Название модели
        self.model = llm_config.get('model_name', 'google/gemma-3-4b')
        
        # Максимальное количество токенов в ответе
        self.max_tokens = llm_config.get('max_tokens', 2048)
        
        # Температура генерации
        self.temperature = llm_config.get('temperature', 0.7)
        
        # Инициализация клиента OpenAI
        # Примечание: Используем OpenAI SDK 2.x с кастомным base_url для локальной модели
        try:
            self.client = OpenAI(
                api_key=self.api_key if self.api_key else "not-needed",
                base_url=self.base_url
            )
            self._available = True
            logger.info(f"LLM инициализирован: модель={self.model}, base_url={self.base_url}")
        except Exception as e:
            logger.error(f"Ошибка инициализации LLM клиента: {e}")
            self.client = None
            self._available = False

    def analyze_text(self, text: str) -> Optional[str]:
        """
        Анализирует текст и выделяет ключевые моменты.

        Args:
            text (str): Текст для анализа.

        Returns:
            str: Анализ текста или None в случае ошибки.
        """
        if not self._available:
            logger.error("LLM клиент недоступен")
            return None
        
        try:
            prompt = f"""
            Проанализируй следующий текст и выдели ключевые моменты.
            Ответь кратко и структурированно на русском языке.
            
            Текст: {text}
            
            Анализ:
            """
            
            # Используем OpenAI SDK 2.x API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"LLM анализ завершен: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при анализе LLM: {e}")
            return None

    def generate_response(self, user_message: str, context: str = "", gender: str = "female", name: str = "", custom_system_prompt: str = "", custom_max_tokens: int = 0) -> Optional[str]:
        """
        Генерирует ответ на сообщение пользователя.

        Args:
            user_message (str): Сообщение пользователя.
            context (str): Контекст разговора в формате "Пользователь: ...\nАссистент: ..."
            gender (str): Пол помощника - "female" или "male"
            name (str): Имя помощника
            custom_system_prompt (str): Кастомный системный промт (если указан, используется вместо стандартного)
            custom_max_tokens (int): Кастомное количество токенов (если указано, используется вместо стандартного)

        Returns:
            str: Сгенерированный ответ или None в случае ошибки.
        """
        if not self._available:
            logger.error("LLM клиент недоступен")
            return None
        
        try:
            # Примечание: если передан кастомный промт, используем его
            if custom_system_prompt:
                # Добавляем базовые правила к кастомному промту
                assistant_name = name if name else ("Дима" if gender == "male" else "Маша")
                gender_rules = "мужской род (я рад, я готов)" if gender == "male" else "женский род (я рада, я готова)"
                system_prompt = f"""{custom_system_prompt}

Дополнительно:
- Тебя зовут {assistant_name}
- Используй {gender_rules}
- Отвечай на русском языке
- Если пользователь поздоровался - ответь на приветствие кратко, но затем переходи к сути
- НЕ здоровайся и НЕ представляйся сам, если пользователь не поздоровался - отвечай сразу по существу"""
            elif gender == "male":
                assistant_name = name if name else "Дима"
                system_prompt = f"""Ты голосовой помощник. Тебя зовут {assistant_name}. Ты мужчина.

СТРОГО: Если тебя спросят как тебя зовут - отвечай ТОЛЬКО "{assistant_name}". Никаких других имён!

Правила:
- Используй мужской род (я рад, я готов, я сделал)
- Отвечай на русском языке кратко (2-4 предложения)
- Отвечай на любые вопросы по существу
- НЕ придумывай себе другие имена
- НЕ используй эмодзи и markdown
- Если пользователь поздоровался (привет, здравствуй и т.д.) - ответь на приветствие кратко, но затем переходи к сути
- НЕ здоровайся и НЕ представляйся сам, если пользователь не поздоровался - отвечай сразу по существу вопроса
- НЕ начинай каждый ответ с "Здравствуйте! Я – {assistant_name}, ваш голосовой помощник" - это избыточно"""
            else:
                assistant_name = name if name else "помощница"
                system_prompt = f"""Ты голосовой помощник. Тебя зовут {assistant_name}. Ты женщина.

СТРОГО: Если тебя спросят как тебя зовут - отвечай ТОЛЬКО "{assistant_name}". Никаких других имён!

Правила:
- Используй женский род (я рада, я готова, я сделала)
- Отвечай на русском языке кратко (2-4 предложения)
- Отвечай на любые вопросы по существу
- НЕ придумывай себе другие имена
- НЕ используй эмодзи и markdown
- Если пользователь поздоровался (привет, здравствуй и т.д.) - ответь на приветствие кратко, но затем переходи к сути
- НЕ здоровайся и НЕ представляйся сам, если пользователь не поздоровался - отвечай сразу по существу вопроса
- НЕ начинай каждый ответ с "Здравствуйте! Я – {assistant_name}, ваш голосовой помощник" - это избыточно"""
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Примечание: контекст должен быть в формате чередующихся user/assistant
            # Парсим контекст если он в формате "Пользователь: ...\nАссистент: ..."
            if context:
                context_lines = context.strip().split('\n')
                for line in context_lines:
                    line = line.strip()
                    if line.startswith('Пользователь:'):
                        messages.append({"role": "user", "content": line[len('Пользователь:'):].strip()})
                    elif line.startswith('Ассистент:'):
                        messages.append({"role": "assistant", "content": line[len('Ассистент:'):].strip()})
            
            messages.append({"role": "user", "content": user_message})
            
            # Примечание: используем кастомные max_tokens если указаны, иначе стандартные
            effective_max_tokens = custom_max_tokens if custom_max_tokens > 0 else self.max_tokens
            
            # Используем OpenAI SDK 2.x API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=effective_max_tokens
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"LLM ответ сгенерирован: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при генерации ответа LLM: {e}")
            return None

    def generate_response_stream(self, user_message: str, context: str = "", gender: str = "female", name: str = "", custom_system_prompt: str = "", custom_max_tokens: int = 0):
        """
        Генерирует ответ на сообщение пользователя в потоковом режиме.
        
        Args:
            user_message (str): Сообщение пользователя.
            context (str): Контекст разговора в формате "Пользователь: ...\nАссистент: ..."
            gender (str): Пол помощника - "female" или "male"
            name (str): Имя помощника
            custom_system_prompt (str): Кастомный системный промт (если указан, используется вместо стандартного)
            custom_max_tokens (int): Кастомное количество токенов (если указано, используется вместо стандартного)
            
        Yields:
            str: Части ответа по мере генерации.
        """
        if not self._available:
            logger.error("LLM клиент недоступен")
            return
        
        try:
            # Примечание: если передан кастомный промт, используем его
            if custom_system_prompt:
                # Добавляем базовые правила к кастомному промту
                assistant_name = name if name else ("Дима" if gender == "male" else "Маша")
                gender_rules = "мужской род (я рад, я готов)" if gender == "male" else "женский род (я рада, я готова)"
                system_prompt = f"""{custom_system_prompt}

Дополнительно:
- Тебя зовут {assistant_name}
- Используй {gender_rules}
- Отвечай на русском языке
- Если пользователь поздоровался - ответь на приветствие кратко, но затем переходи к сути
- НЕ здоровайся и НЕ представляйся сам, если пользователь не поздоровался - отвечай сразу по существу"""
            elif gender == "male":
                assistant_name = name if name else "Дима"
                system_prompt = f"""Ты голосовой помощник. Тебя зовут {assistant_name}. Ты мужчина.

СТРОГО: Если тебя спросят как тебя зовут - отвечай ТОЛЬКО "{assistant_name}". Никаких других имён!

Правила:
- Используй мужской род (я рад, я готов, я сделал)
- Отвечай на русском языке кратко (2-4 предложения)
- Отвечай на любые вопросы по существу
- НЕ придумывай себе другие имена
- НЕ используй эмодзи и markdown
- Если пользователь поздоровался (привет, здравствуй и т.д.) - ответь на приветствие кратко, но затем переходи к сути
- НЕ здоровайся и НЕ представляйся сам, если пользователь не поздоровался - отвечай сразу по существу вопроса
- НЕ начинай каждый ответ с "Здравствуйте! Я – {assistant_name}, ваш голосовой помощник" - это избыточно"""
            else:
                assistant_name = name if name else "помощница"
                system_prompt = f"""Ты голосовой помощник. Тебя зовут {assistant_name}. Ты женщина.

СТРОГО: Если тебя спросят как тебя зовут - отвечай ТОЛЬКО "{assistant_name}". Никаких других имён!

Правила:
- Используй женский род (я рада, я готова, я сделала)
- Отвечай на русском языке кратко (2-4 предложения)
- Отвечай на любые вопросы по существу
- НЕ придумывай себе другие имена
- НЕ используй эмодзи и markdown
- Если пользователь поздоровался (привет, здравствуй и т.д.) - ответь на приветствие кратко, но затем переходи к сути
- НЕ здоровайся и НЕ представляйся сам, если пользователь не поздоровался - отвечай сразу по существу вопроса
- НЕ начинай каждый ответ с "Здравствуйте! Я – {assistant_name}, ваш голосовой помощник" - это избыточно"""
            
            messages = [{"role": "system", "content": system_prompt}]
            
            # Примечание: парсим контекст в формате "Пользователь: ...\nАссистент: ..."
            if context:
                context_lines = context.strip().split('\n')
                for line in context_lines:
                    line = line.strip()
                    if line.startswith('Пользователь:'):
                        messages.append({"role": "user", "content": line[len('Пользователь:'):].strip()})
                    elif line.startswith('Ассистент:'):
                        messages.append({"role": "assistant", "content": line[len('Ассистент:'):].strip()})
            
            messages.append({"role": "user", "content": user_message})
            
            # Примечание: используем кастомные max_tokens если указаны, иначе стандартные
            effective_max_tokens = custom_max_tokens if custom_max_tokens > 0 else self.max_tokens
            
            # Используем OpenAI SDK 2.x API с потоковой передачей
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=effective_max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Ошибка при потоковой генерации ответа LLM: {e}")

    def extract_keywords(self, text: str) -> List[str]:
        """
        Извлекает ключевые слова из текста.
        
        Args:
            text (str): Текст для анализа.
            
        Returns:
            List[str]: Список ключевых слов.
        """
        if not self._available:
            logger.error("LLM клиент недоступен")
            return []
        
        try:
            prompt = f"""
            Извлеки ключевые слова из следующего текста.
            Верни только слова через запятую, без дополнительных объяснений.
            
            Текст: {text}
            
            Ключевые слова:
            """
            
            # Используем OpenAI SDK 2.x API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=100
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            logger.info(f"Извлечены ключевые слова: {keywords}")
            return keywords
            
        except Exception as e:
            logger.error(f"Ошибка при извлечении ключевых слов: {e}")
            return []

    def summarize_text(self, text: str) -> str:
        """
        Составляет краткое описание текста.
        
        Args:
            text (str): Текст для анализа.
            
        Returns:
            str: Краткое описание текста.
        """
        if not self._available:
            logger.error("LLM клиент недоступен")
            return ""
        
        try:
            prompt = f"""
            Составь краткое описание следующего текста на русском языке.
            Максимум 2-3 предложения.
            
            Текст: {text}
            
            Краткое описание:
            """
            
            # Используем OpenAI SDK 2.x API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"Создано краткое описание: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Ошибка при создании краткого описания: {e}")
            return ""

    def is_available(self) -> bool:
        """
        Проверяет доступность LLM сервиса.
        
        Returns:
            bool: True если сервис доступен, False иначе.
        """
        return self._available
    
    def test_connection(self) -> bool:
        """
        Тестирует соединение с LLM сервером.
        
        Returns:
            bool: True если соединение успешно, False иначе.
        """
        if not self._available:
            return False
        
        try:
            # Простой тестовый запрос
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Привет! Ответь одним словом."}],
                temperature=0.1,
                max_tokens=10
            )
            
            if response.choices and response.choices[0].message.content:
                logger.info("Тестовое соединение с LLM успешно")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Ошибка тестового соединения с LLM: {e}")
            return False
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Получает список доступных моделей из LLM сервера.
        
        Returns:
            List[Dict]: Список моделей с информацией о каждой.
        """
        if not self._available:
            return []
        
        try:
            # Запрос списка моделей через OpenAI-совместимый API
            models_response = self.client.models.list()
            
            models = []
            for model in models_response.data:
                models.append({
                    "id": model.id,
                    "object": getattr(model, 'object', 'model'),
                    "owned_by": getattr(model, 'owned_by', 'unknown')
                })
            
            logger.info(f"Получено {len(models)} доступных моделей")
            return models
            
        except Exception as e:
            logger.error(f"Ошибка получения списка моделей: {e}")
            return []
    
    def set_model(self, model_id: str) -> bool:
        """
        Устанавливает текущую модель для использования.
        
        Args:
            model_id (str): Идентификатор модели.
            
        Returns:
            bool: True если модель успешно установлена, False иначе.
        """
        if not self._available:
            return False
        
        try:
            old_model = self.model
            self.model = model_id
            logger.info(f"Модель изменена: {old_model} -> {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка установки модели: {e}")
            return False
    
    def get_current_model(self) -> str:
        """
        Возвращает текущую используемую модель.
        
        Returns:
            str: Идентификатор текущей модели.
        """
        return self.model
