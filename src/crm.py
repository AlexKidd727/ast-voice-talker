# Voice Talker - Модуль CRM системы
# Исправлено: совместимость с новой структурой базы данных

import sqlite3
import logging
import requests
from typing import Optional, Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

def create_ticket(customer_data: Dict[str, Any], needs: List[str], db_path: str = "data.db") -> Optional[int]:
    """
    Создает заявку в CRM системе на основе данных о клиенте и его потребностях.

    Args:
        customer_data (dict): Данные о клиенте (имя, телефон, email и т.д.).
        needs (list): Список потребностей клиента.
        db_path (str): Путь к базе данных.

    Returns:
        int: ID созданной заявки или None в случае ошибки.
    """
    try:
        # Создаем директорию если не существует
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Создаем таблицу tickets если не существует
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

        # Формируем SQL запрос для создания заявки
        sql = """
        INSERT INTO tickets (customer_name, customer_phone, customer_email, needs, status)
            VALUES (?, ?, ?, ?, ?)
        """

        # Подготавливаем данные для вставки
        needs_str = ", ".join(needs) if isinstance(needs, list) else str(needs)
        data = (
            customer_data.get("name", ""),
            customer_data.get("phone", ""),
            customer_data.get("email", ""),
            needs_str,
            "new",  # Начальный статус заявки
        )

        cursor.execute(sql, data)
        conn.commit()

        ticket_id = cursor.lastrowid
        logger.info(f"Заявка успешно создана с ID: {ticket_id}")
        return ticket_id

    except sqlite3.Error as e:
        logger.error(f"Ошибка при создании заявки: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def get_ticket_by_id(ticket_id: int, db_path: str = "data.db") -> Optional[Dict[str, Any]]:
    """
    Получает заявку по ID.

    Args:
        ticket_id (int): ID заявки.
        db_path (str): Путь к базе данных.

    Returns:
        dict: Данные заявки или None если не найдена.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None

    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении заявки {ticket_id}: {e}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def update_ticket_status(ticket_id: int, status: str, db_path: str = "data.db") -> bool:
    """
    Обновляет статус заявки.

    Args:
        ticket_id (int): ID заявки.
        status (str): Новый статус.
        db_path (str): Путь к базе данных.

    Returns:
        bool: True если успешно, False иначе.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE tickets 
            SET status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (status, ticket_id))
        
        conn.commit()
        logger.info(f"Статус заявки {ticket_id} обновлен на: {status}")
        return True

    except sqlite3.Error as e:
        logger.error(f"Ошибка при обновлении статуса заявки {ticket_id}: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def get_all_tickets(db_path: str = "data.db") -> List[Dict[str, Any]]:
    """
    Получает все заявки.

    Args:
        db_path (str): Путь к базе данных.

    Returns:
        list: Список заявок.
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM tickets ORDER BY created_at DESC")
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]

    except sqlite3.Error as e:
        logger.error(f"Ошибка при получении заявок: {e}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def send_to_external_crm(customer_data: Dict[str, Any], needs: List[str], config: Dict[str, Any]) -> Optional[str]:
    """
    Отправляет данные во внешнюю CRM систему.

    Args:
        customer_data (dict): Данные клиента.
        needs (list): Потребности клиента.
        config (dict): Конфигурация с настройками CRM.

    Returns:
        str: ID заявки во внешней системе или None.
    """
    try:
        crm_config = config.get('crm', {})
        api_endpoint = crm_config.get('api_endpoint', '')
        auth_token = crm_config.get('auth_token', '')
        
        if not api_endpoint or not auth_token:
            logger.warning("CRM API не настроен, создаем локальную заявку")
            return create_ticket(customer_data, needs)
        
        # Подготовка данных для отправки
        payload = {
            "customer": customer_data,
            "needs": needs,
            "source": "voice_talker"
        }
        
        headers = {
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            api_endpoint,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            external_id = result.get('id', result.get('ticket_id'))
            logger.info(f"Заявка отправлена во внешнюю CRM с ID: {external_id}")
            return str(external_id)
        else:
            logger.error(f"Ошибка отправки в CRM: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        logger.error(f"Ошибка при отправке во внешнюю CRM: {e}")
        return None
