# Voice Talker - Скрипт запуска сервера

#
# Использование:
#   python run.py              - запуск на порту 8000
#   python run.py --port 8080  - запуск на порту 8080
#   python run.py --reload     - запуск с автоперезагрузкой (для разработки)

import argparse
import os
import sys

# Добавляем папку src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def main():
    parser = argparse.ArgumentParser(description='Voice Talker Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host для запуска (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8300, help='Порт для запуска (default: 8300)')
    parser.add_argument('--reload', action='store_true', help='Автоперезагрузка при изменении файлов')
    parser.add_argument('--workers', type=int, default=1, help='Количество воркеров (default: 1)')
    args = parser.parse_args()
    
    import uvicorn
    
    print("=" * 60)
    print("  Voice Talker - Голосовой помощник с AI")
    print("=" * 60)
    print(f"  Сервер: http://{args.host}:{args.port}")
    print(f"  Документация API: http://{args.host}:{args.port}/docs")
    print(f"  LLM сервер: http://192.168.1.250:1234")
    print(f"  Модель: google/gemma-3-4b")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info"
    )


if __name__ == "__main__":
    main()
