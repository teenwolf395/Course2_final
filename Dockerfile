FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Копирование исходного кода
COPY . .

# Создание пользователя для безопасности
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Экспорт порта
EXPOSE 8000

# Переменные окружения по умолчанию
ENV MODEL_PATH=/app/model \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    LOG_LEVEL=INFO \
    MAX_LENGTH=200

# Команда запуска
CMD ["python", "api_deploy.py"]