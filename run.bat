@echo off
REM Voice Talker - Скрипт запуска для Windows

echo ============================================================
echo   Voice Talker - Голосовой помощник с AI
echo ============================================================

cd /d "%~dp0"

REM Проверка виртуального окружения
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo   Виртуальное окружение активировано
) else (
    echo   Виртуальное окружение не найдено, используется системный Python
)

echo   Запуск сервера...
echo ============================================================

cd src
python -m uvicorn app:app --host 0.0.0.0 --port 8300 --reload

pause
