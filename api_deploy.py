#!/usr/bin/env python3
"""
ALBERT Intent Classifier API
Развертывание обученной модели ALBERT как REST API сервиса
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
from transformers import AlbertForSequenceClassification, AlbertTokenizer

# ==================== КОНФИГУРАЦИЯ ====================
# Конфигурация через переменные окружения (оптимально для Docker)

# Пути (можно переопределить через переменные окружения)
MODEL_PATH = os.getenv("MODEL_PATH", "./training_results/best_model")
LOG_DIR = os.getenv("LOG_DIR", "./logs")
CONFIG_FILE = os.getenv("CONFIG_FILE", "./api_config.json")
# Путь к файлу с оригинальными названиями классов
LABEL_MAPPING_FILE = os.getenv("LABEL_MAPPING_FILE", "./label_mapping.json")

# Параметры API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_DEBUG = os.getenv("API_DEBUG", "false").lower() == "true"

# Параметры модели
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "200"))
BATCH_SIZE_LIMIT = int(os.getenv("BATCH_SIZE_LIMIT", "100"))

# Настройки логирования
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_TO_FILE = os.getenv("LOG_TO_FILE", "true").lower() == "true"

# CORS настройки
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# ==================== НАСТРОЙКА ЛОГИРОВАНИЯ ====================

# Создаем директорию для логов
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# Формат логов
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
date_format = "%Y-%m-%d %H:%M:%S"

# Конфигурация логирования
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=log_format,
    datefmt=date_format
)

logger = logging.getLogger(__name__)

# Добавляем файловый обработчик если нужно
if LOG_TO_FILE:
    log_file = Path(LOG_DIR) / f"api_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logger.addHandler(file_handler)

# ==================== МОДЕЛЬ ДАННЫХ PYDANTIC ====================

class HealthResponse(BaseModel):
    """Ответ для проверки здоровья API"""
    status: str = Field(..., description="Статус сервиса")
    model_loaded: bool = Field(..., description="Модель загружена")
    timestamp: float = Field(..., description="Временная метка")
    version: str = Field(..., description="Версия API")

class ModelInfoResponse(BaseModel):
    """Информация о модели"""
    model_name: str = Field(..., description="Имя модели")
    model_type: str = Field(..., description="Тип модели")
    num_labels: int = Field(..., description="Количество классов")
    max_length: int = Field(..., description="Максимальная длина текста")
    device: str = Field(..., description="Устройство выполнения")
    loaded_at: str = Field(..., description="Время загрузки")

class PredictionRequest(BaseModel):
    """Запрос для предсказания одного текста"""
    text: str = Field(..., min_length=1, max_length=MAX_LENGTH, 
                      description=f"Текст для классификации (до {MAX_LENGTH} символов)")

class BatchPredictionRequest(BaseModel):
    """Запрос для пакетного предсказания"""
    texts: List[str] = Field(..., description=f"Список текстов для классификации (максимум {BATCH_SIZE_LIMIT})")
    
    class Config:
        schema_extra = {
            "example": {
                "texts": [
                    "How do I activate my credit card?",
                    "I need to transfer money to another account"
                ]
            }
        }

class ClassPrediction(BaseModel):
    """Предсказание для одного класса"""
    class_id: int = Field(..., description="ID класса")
    class_name: Optional[str] = Field(None, description="Название класса")
    probability: float = Field(..., ge=0, le=1, description="Вероятность")
    confidence: float = Field(..., ge=0, le=1, description="Уверенность")

class PredictionResponse(BaseModel):
    """Ответ с предсказанием"""
    text: str = Field(..., description="Исходный текст")
    predicted_class: ClassPrediction = Field(..., description="Предсказанный класс")
    all_predictions: List[ClassPrediction] = Field(..., description="Все предсказания")
    processing_time_ms: float = Field(..., description="Время обработки в миллисекундах")

class BatchPredictionResponse(BaseModel):
    """Ответ для пакетного предсказания"""
    predictions: List[PredictionResponse] = Field(..., description="Список предсказаний")
    total_time_ms: float = Field(..., description="Общее время обработки")
    average_time_ms: float = Field(..., description="Среднее время на запрос")

class ErrorResponse(BaseModel):
    """Ответ при ошибке"""
    error: str = Field(..., description="Сообщение об ошибке")
    detail: Optional[str] = Field(None, description="Детали ошибки")
    timestamp: float = Field(..., description="Временная метка")

# ==================== МЕНЕДЖЕР МОДЕЛИ ====================

class ModelManager:
    """Класс для управления моделью ALBERT"""
    
    def __init__(self, model_path: str, label_mapping_file: str = None):
        self.model_path = Path(model_path)
        self.label_mapping_file = Path(label_mapping_file) if label_mapping_file else None
        self.model = None
        self.tokenizer = None
        self.device = None
        self.metadata = {}
        self.class_names = {}  # Здесь будут оригинальные названия классов
        self.loaded_at = None
        
    def load_model(self):
        """Загрузка модели и токенизатора"""
        try:
            logger.info(f"Загрузка модели из {self.model_path}")
            
            if not self.model_path.exists():
                raise FileNotFoundError(f"Путь к модели не существует: {self.model_path}")
            
            # Загрузка модели
            self.model = AlbertForSequenceClassification.from_pretrained(
                str(self.model_path)
            )
            
            # Загрузка токенизатора
            self.tokenizer = AlbertTokenizer.from_pretrained(
                str(self.model_path)
            )
            
            # Определение устройства
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            # Загрузка метаданных
            metadata_path = self.model_path / "model_info.json"
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            
            # ПРИОРИТЕТ 1: Загружаем оригинальные названия классов из label_mapping.json
            if self.label_mapping_file and self.label_mapping_file.exists():
                try:
                    with open(self.label_mapping_file, 'r', encoding='utf-8') as f:
                        label_mapping_data = json.load(f)
                    
                    # Проверяем структуру файла
                    if isinstance(label_mapping_data, dict):
                        # Если файл содержит поля label2id и id2label
                        if 'id2label' in label_mapping_data:
                            id2label = label_mapping_data['id2label']
                            # Преобразуем строковые ключи в int
                            self.class_names = {}
                            for key, value in id2label.items():
                                try:
                                    class_id = int(key)
                                    self.class_names[class_id] = value
                                except ValueError:
                                    logger.warning(f"Некорректный ID класса в файле: {key}")
                            
                            logger.info(f"Загружены оригинальные названия классов из id2label: {len(self.class_names)} классов")
                            logger.info(f"Классы: {list(self.class_names.values())}")
                        
                        elif 'label2id' in label_mapping_data:
                            # Если есть только label2id, создаем обратный маппинг
                            label2id = label_mapping_data['label2id']
                            self.class_names = {}
                            for label_name, label_id in label2id.items():
                                if isinstance(label_id, int):
                                    self.class_names[label_id] = label_name
                            
                            logger.info(f"Загружены оригинальные названия классов из label2id: {len(self.class_names)} классов")
                            logger.info(f"Классы: {list(label2id.keys())}")
                        else:
                            logger.warning(f"Неожиданный формат файла {self.label_mapping_file}")
                    else:
                        logger.warning(f"Файл {self.label_mapping_file} должен содержать словарь")
                        
                except Exception as e:
                    logger.error(f"Ошибка загрузки файла label_mapping.json: {e}")
                    import traceback
                    traceback.print_exc()
            
            # ПРИОРИТЕТ 2: Если нет отдельного файла, проверяем в папке модели
            if not self.class_names:
                label_mapping_in_model = self.model_path / "label_mapping.json"
                if label_mapping_in_model.exists():
                    try:
                        with open(label_mapping_in_model, 'r', encoding='utf-8') as f:
                            label_mapping_data = json.load(f)
                        
                        if isinstance(label_mapping_data, dict) and 'id2label' in label_mapping_data:
                            id2label = label_mapping_data['id2label']
                            self.class_names = {}
                            for key, value in id2label.items():
                                try:
                                    class_id = int(key)
                                    self.class_names[class_id] = value
                                except ValueError:
                                    continue
                            
                            logger.info(f"Загружены названия классов из файла модели: {len(self.class_names)} классов")
                    except Exception as e:
                        logger.error(f"Ошибка загрузки label_mapping.json из модели: {e}")
            
            # ПРИОРИТЕТ 3: Если все еще нет, проверяем id2label.json в папке модели
            if not self.class_names:
                id2label_file = self.model_path / "id2label.json"
                if id2label_file.exists():
                    try:
                        with open(id2label_file, 'r', encoding='utf-8') as f:
                            id2label_data = json.load(f)
                        
                        # Преобразуем строковые ключи в int
                        self.class_names = {}
                        for key, value in id2label_data.items():
                            try:
                                class_id = int(key)
                                self.class_names[class_id] = value
                            except:
                                continue
                        
                        logger.info(f"Загружены названия классов из id2label.json: {len(self.class_names)} классов")
                    except Exception as e:
                        logger.error(f"Ошибка загрузки id2label.json: {e}")
            
            # ПРИОРИТЕТ 4: Если все еще нет, проверяем в конфиге модели
            if not self.class_names and hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                self.class_names = self.model.config.id2label
                logger.info(f"Загружены названия классов из конфига модели: {len(self.class_names)} классов")
                
                # Проверяем, не являются ли это общими названиями (LABEL_0, LABEL_1, etc)
                if self.class_names and list(self.class_names.values())[0].startswith("LABEL_"):
                    logger.warning("Обнаружены общие названия классов (LABEL_X)")
            
            # ПРИОРИТЕТ 5: Если все еще нет, создаем названия на основе ID
            if not self.class_names:
                actual_num_labels = self.model.config.num_labels
                self.class_names = {i: f"class_{i}" for i in range(actual_num_labels)}
                logger.info(f"Созданы названия классов по умолчанию для {actual_num_labels} классов")
            
            # Проверяем количество классов
            actual_num_labels = self.model.config.num_labels
            loaded_num_classes = len(self.class_names)
            
            logger.info(f"Модель имеет {actual_num_labels} классов")
            logger.info(f"Загружено {loaded_num_classes} названий классов")
            
            # Создаем окончательный список названий классов
            final_class_names = {}
            for i in range(actual_num_labels):
                if i in self.class_names:
                    final_class_names[i] = self.class_names[i]
                else:
                    # Проверяем, есть ли название в виде строки ключа
                    str_key = str(i)
                    if str_key in self.class_names:
                        final_class_names[i] = self.class_names[str_key]
                    else:
                        final_class_names[i] = f"class_{i}"
            
            self.class_names = final_class_names
            
            self.loaded_at = datetime.now().isoformat()
            
            logger.info(f"Модель загружена успешно!")
            logger.info(f"   Устройство: {self.device}")
            logger.info(f"   Количество классов: {actual_num_labels}")
            logger.info(f"   Модель: {self.get_model_name()}")
            logger.info(f"   Названия классов:")
            for i in range(min(actual_num_labels, 13)):  # Показываем все 13 классов
                class_name = self.class_names.get(i, f"class_{i}")
                logger.info(f"      {i}: {class_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def get_model_name(self) -> str:
        """Получение имени модели"""
        return self.metadata.get('model_name', 'ALBERT Intent Classifier')
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        actual_num_labels = self.model.config.num_labels if self.model else 0
        
        # Берем только те классы, которые существуют в модели
        class_names_sample = {}
        for i in range(min(13, actual_num_labels)):  # Показываем все 13 классов
            class_names_sample[i] = self.class_names.get(i, f"class_{i}")
        
        return {
            "model_name": self.get_model_name(),
            "model_type": "ALBERT",
            "num_labels": actual_num_labels,
            "max_length": MAX_LENGTH,
            "device": str(self.device),
            "loaded_at": self.loaded_at,
            "metadata": self.metadata,
            "class_names_sample": class_names_sample,
            "all_class_names": {k: self.class_names.get(k, f"class_{k}") for k in range(actual_num_labels)}
        }
    
    def get_class_name(self, class_id: int) -> str:
        """Получение названия класса по ID"""
        # Проверяем, существует ли такой класс в модели
        if self.model and class_id >= self.model.config.num_labels:
            logger.warning(f"Запрошен несуществующий класс ID: {class_id} (максимум: {self.model.config.num_labels-1})")
            return f"unknown_class_{class_id}"
        
        # Пробуем получить название класса
        class_name = self.class_names.get(class_id, f"class_{class_id}")
        
        # Если это общее название (LABEL_X), пытаемся найти лучшее
        if isinstance(class_name, str) and class_name.startswith("LABEL_"):
            # Пытаемся найти в виде строкового ключа
            str_key = str(class_id)
            if str_key in self.class_names and not self.class_names[str_key].startswith("LABEL_"):
                return self.class_names[str_key]
            
            # Если не нашли, возвращаем как есть
            return class_name
        
        return class_name
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """Предсказание для одного текста"""
        start_time = time.time()
        
        try:
            # Токенизация
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            
            # Перенос на устройство
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**encoding)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                probabilities = probs[0].cpu().numpy()
            
            # Проверяем количество вероятностей
            num_probabilities = len(probabilities)
            expected_num_labels = self.model.config.num_labels
            
            if num_probabilities != expected_num_labels:
                logger.warning(f"Несоответствие количества вероятностей: {num_probabilities} vs {expected_num_labels}")
                # Ограничиваем до минимального количества
                num_to_use = min(num_probabilities, expected_num_labels)
                probabilities = probabilities[:num_to_use]
            
            # Получение топ-1 предсказания
            predicted_idx = np.argmax(probabilities)
            confidence = float(probabilities[predicted_idx])
            
            # Получаем название предсказанного класса
            predicted_class_name = self.get_class_name(predicted_idx)
            
            # Подготовка всех предсказаний
            all_predictions = []
            for i, prob in enumerate(probabilities):
                # Проверяем, что класс существует в модели
                if i < expected_num_labels:
                    class_name = self.get_class_name(i)
                    all_predictions.append({
                        "class_id": i,
                        "class_name": class_name,
                        "probability": float(prob),
                        "confidence": float(prob)
                    })
            
            # Сортировка по убыванию вероятности
            all_predictions.sort(key=lambda x: x["probability"], reverse=True)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            result = {
                "text": text,
                "predicted_class": {
                    "class_id": int(predicted_idx),
                    "class_name": predicted_class_name,
                    "probability": confidence,
                    "confidence": confidence
                },
                "all_predictions": all_predictions,
                "processing_time_ms": processing_time_ms
            }
            
            logger.debug(f"Предсказание: '{text[:50]}...' -> {predicted_class_name} (ID: {predicted_idx}, {confidence:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка предсказания для текста '{text[:50]}...': {e}")
            raise
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Пакетное предсказание"""
        start_time = time.time()
        results = []
        
        try:
            # Проверка лимита
            if len(texts) > BATCH_SIZE_LIMIT:
                raise ValueError(f"Слишком много текстов. Максимум: {BATCH_SIZE_LIMIT}")
            
            # Токенизация батча
            encoding = self.tokenizer(
                texts,
                truncation=True,
                padding=True,
                max_length=MAX_LENGTH,
                return_tensors="pt"
            )
            
            # Перенос на устройство
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            
            # Предсказание
            with torch.no_grad():
                outputs = self.model(**encoding)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                all_probabilities = probs.cpu().numpy()
            
            expected_num_labels = self.model.config.num_labels
            
            # Обработка каждого текста
            for i, text in enumerate(texts):
                probabilities = all_probabilities[i]
                
                # Проверяем количество вероятностей
                num_probabilities = len(probabilities)
                if num_probabilities != expected_num_labels:
                    # Ограничиваем до минимального количества
                    num_to_use = min(num_probabilities, expected_num_labels)
                    probabilities = probabilities[:num_to_use]
                
                predicted_idx = np.argmax(probabilities)
                confidence = float(probabilities[predicted_idx])
                
                # Получаем название предсказанного класса
                predicted_class_name = self.get_class_name(predicted_idx)
                
                # Подготовка всех предсказаний
                text_predictions = []
                for j, prob in enumerate(probabilities):
                    # Проверяем, что класс существует в модели
                    if j < expected_num_labels:
                        class_name = self.get_class_name(j)
                        text_predictions.append({
                            "class_id": j,
                            "class_name": class_name,
                            "probability": float(prob),
                            "confidence": float(prob)
                        })
                
                # Сортировка
                text_predictions.sort(key=lambda x: x["probability"], reverse=True)
                
                results.append({
                    "text": text,
                    "predicted_class": {
                        "class_id": int(predicted_idx),
                        "class_name": predicted_class_name,
                        "probability": confidence,
                        "confidence": confidence
                    },
                    "all_predictions": text_predictions,
                    "processing_time_ms": 0  # Заполним позже
                })
            
            # Вычисление времени
            total_time_ms = (time.time() - start_time) * 1000
            avg_time_ms = total_time_ms / len(texts) if texts else 0
            
            # Установка времени обработки
            for result in results:
                result["processing_time_ms"] = avg_time_ms
            
            # Логирование результатов
            logger.info(f"Пакетное предсказание: {len(texts)} текстов за {total_time_ms:.2f}ms")
            for i, result in enumerate(results[:3]):  # Логируем только первые 3
                pred = result['predicted_class']
                logger.info(f"  Текст {i+1}: '{texts[i][:40]}...' -> {pred['class_name']} ({pred['probability']:.2%})")
            
            return {
                "predictions": results,
                "total_time_ms": total_time_ms,
                "average_time_ms": avg_time_ms
            }
            
        except Exception as e:
            logger.error(f"Ошибка пакетного предсказания: {e}")
            raise

# ==================== ИНИЦИАЛИЗАЦИЯ ====================

# Создаем менеджер модели
model_manager = ModelManager(MODEL_PATH, LABEL_MAPPING_FILE)

# ==================== FASTAPI ПРИЛОЖЕНИЕ ====================

app = FastAPI(
    title="ALBERT Intent Classifier API",
    description="REST API для классификации банковских интентов с помощью обученной модели ALBERT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Пропускаем статические файлы
    if request.url.path.startswith("/static/"):
        response = await call_next(request)
        return response
    
    # Логируем запрос
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"Входящий запрос: {client_host} - {request.method} {request.url.path}")
    
    # Обрабатываем запрос
    response = await call_next(request)
    
    # Логируем ответ
    process_time = (time.time() - start_time) * 1000
    logger.info(f"Исходящий ответ: {client_host} - {request.method} {request.url.path} - {response.status_code} - {process_time:.2f}ms")
    
    # Добавляем время обработки в заголовки
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    
    return response

# ==================== ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с веб-интерфейсом"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ALBERT Intent Classifier</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            
            .header {
                text-align: center;
                margin-bottom: 40px;
                color: white;
            }
            
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.2rem;
                opacity: 0.9;
            }
            
            .card {
                background: white;
                border-radius: 15px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            
            .card h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.5rem;
            }
            
            .tabs {
                display: flex;
                gap: 10px;
                margin-bottom: 20px;
                border-bottom: 2px solid #eee;
            }
            
            .tab {
                padding: 10px 20px;
                background: none;
                border: none;
                font-size: 1rem;
                cursor: pointer;
                color: #666;
                position: relative;
            }
            
            .tab.active {
                color: #667eea;
                font-weight: bold;
            }
            
            .tab.active::after {
                content: '';
                position: absolute;
                bottom: -2px;
                left: 0;
                right: 0;
                height: 3px;
                background: #667eea;
                border-radius: 3px;
            }
            
            .tab-content {
                display: none;
            }
            
            .tab-content.active {
                display: block;
            }
            
            textarea {
                width: 100%;
                min-height: 120px;
                padding: 15px;
                border: 2px solid #ddd;
                border-radius: 10px;
                font-size: 1rem;
                resize: vertical;
                margin-bottom: 15px;
            }
            
            textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            
            .batch-input {
                margin-bottom: 15px;
            }
            
            .batch-input textarea {
                min-height: 200px;
            }
            
            .button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 25px;
                font-size: 1rem;
                cursor: pointer;
                transition: transform 0.2s;
                margin-right: 10px;
            }
            
            .button:hover {
                transform: translateY(-2px);
            }
            
            .button.secondary {
                background: #6c757d;
            }
            
            .result {
                margin-top: 20px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
                border-left: 4px solid #667eea;
            }
            
            .prediction-item {
                background: white;
                padding: 15px;
                margin-bottom: 10px;
                border-radius: 8px;
                border: 1px solid #eee;
            }
            
            .prediction-header {
                display: flex;
                justify-content: space-between;
                margin-bottom: 10px;
                padding-bottom: 10px;
                border-bottom: 1px solid #eee;
            }
            
            .confidence-bar {
                height: 10px;
                background: #e9ecef;
                border-radius: 5px;
                margin: 10px 0;
                overflow: hidden;
            }
            
            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                border-radius: 5px;
                transition: width 0.3s;
            }
            
            .class-name {
                font-weight: bold;
                color: #333;
            }
            
            .class-probability {
                color: #666;
            }
            
            .top-class {
                background: #e8f4fd;
                border-color: #2196f3;
            }
            
            .status {
                display: inline-block;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.9rem;
                font-weight: bold;
            }
            
            .status.healthy {
                background: #d4edda;
                color: #155724;
            }
            
            .status.error {
                background: #f8d7da;
                color: #721c24;
            }
            
            .loading {
                text-align: center;
                padding: 20px;
                color: #666;
            }
            
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 30px;
                height: 30px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .class-info {
                background: #f1f8ff;
                padding: 10px;
                border-radius: 5px;
                margin: 5px 0;
            }
            
            .class-id {
                color: #6c757d;
                font-size: 0.9rem;
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 10px;
                }
                
                .card {
                    padding: 20px;
                }
                
                .header h1 {
                    font-size: 2rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>ALBERT Intent Classifier</h1>
                <p>REST API для классификации банковских интентов</p>
            </div>
            
            <div class="card">
                <div class="tabs">
                    <button class="tab active" onclick="switchTab('single')">Single Prediction</button>
                    <button class="tab" onclick="switchTab('batch')">Batch Prediction</button>
                    <button class="tab" onclick="switchTab('health')">Health Check</button>
                    <button class="tab" onclick="switchTab('classes')">All Classes</button>
                </div>
                
                <div id="single-tab" class="tab-content active">
                    <h2>Single Text Prediction</h2>
                    <textarea id="single-text" placeholder="Enter text for classification... (max 200 characters)"></textarea>
                    <button class="button" onclick="predictSingle()">Predict</button>
                    <button class="button secondary" onclick="clearSingle()">Clear</button>
                    <div id="single-result" class="result"></div>
                </div>
                
                <div id="batch-tab" class="tab-content">
                    <h2>Batch Prediction</h2>
                    <p>Enter multiple texts (one per line, max 100 texts):</p>
                    <div class="batch-input">
                        <textarea id="batch-texts" placeholder="Text 1&#10;Text 2&#10;Text 3..."></textarea>
                    </div>
                    <button class="button" onclick="predictBatch()">Predict All</button>
                    <button class="button secondary" onclick="clearBatch()">Clear</button>
                    <div id="batch-result" class="result"></div>
                </div>
                
                <div id="health-tab" class="tab-content">
                    <h2>Health Status</h2>
                    <div id="health-status">
                        <div class="loading">
                            <div class="spinner"></div>
                            <p>Checking API health...</p>
                        </div>
                    </div>
                </div>
                
                <div id="classes-tab" class="tab-content">
                    <h2>All Classes</h2>
                    <div id="classes-list">
                        <div class="loading">
                            <div class="spinner"></div>
                            <p>Loading classes list...</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h2>System Information</h2>
                <div id="system-info">
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Loading system information...</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Tab switching
            function switchTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Remove active class from all tab buttons
                document.querySelectorAll('.tab').forEach(btn => {
                    btn.classList.remove('active');
                });
                
                // Show selected tab
                document.getElementById(tabName + '-tab').classList.add('active');
                
                // Add active class to clicked button
                event.target.classList.add('active');
                
                // Load data for specific tabs
                if (tabName === 'health') {
                    checkHealth();
                } else if (tabName === 'classes') {
                    loadClassesList();
                }
            }
            
            // Single prediction
            async function predictSingle() {
                const text = document.getElementById('single-text').value.trim();
                if (!text) {
                    alert('Please enter some text!');
                    return;
                }
                
                if (text.length > 200) {
                    alert('Text is too long! Maximum 200 characters.');
                    return;
                }
                
                const resultDiv = document.getElementById('single-result');
                resultDiv.innerHTML = `
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Processing prediction...</p>
                    </div>
                `;
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text: text })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    displaySingleResult(data);
                    
                } catch (error) {
                    resultDiv.innerHTML = `
                        <div class="prediction-item" style="border-color: #dc3545;">
                            <h3>Error</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                }
            }
            
            function displaySingleResult(data) {
                const resultDiv = document.getElementById('single-result');
                let html = `
                    <div class="prediction-header">
                        <h3>Prediction Result</h3>
                        <span class="status healthy">${data.processing_time_ms.toFixed(2)} ms</span>
                    </div>
                    <div class="prediction-item top-class">
                        <h4>Top Prediction:</h4>
                        <div class="class-info">
                            <div class="prediction-header">
                                <span class="class-name">${data.predicted_class.class_name}</span>
                                <span class="class-probability">${(data.predicted_class.probability * 100).toFixed(2)}%</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${data.predicted_class.probability * 100}%"></div>
                            </div>
                            <p><strong>Class ID:</strong> ${data.predicted_class.class_id}</p>
                            <p><strong>Confidence:</strong> ${(data.predicted_class.confidence * 100).toFixed(2)}%</p>
                        </div>
                    </div>
                    <h4>All Predictions (Top 5):</h4>
                `;
                
                // Показываем только топ-5 предсказаний
                data.all_predictions.slice(0, 5).forEach((pred, index) => {
                    const isTop = pred.class_id === data.predicted_class.class_id;
                    html += `
                        <div class="prediction-item ${isTop ? 'top-class' : ''}">
                            <div class="prediction-header">
                                <span class="class-name">${index + 1}. ${pred.class_name}</span>
                                <span class="class-probability">${(pred.probability * 100).toFixed(2)}%</span>
                            </div>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${pred.probability * 100}%"></div>
                            </div>
                            <p class="class-id">ID: ${pred.class_id}</p>
                        </div>
                    `;
                });
                
                if (data.all_predictions.length > 5) {
                    html += `<p style="color: #666; text-align: center;">... and ${data.all_predictions.length - 5} more classes</p>`;
                }
                
                resultDiv.innerHTML = html;
            }
            
            // Batch prediction
            async function predictBatch() {
                const textarea = document.getElementById('batch-texts');
                const texts = textarea.value.trim().split('\\n').filter(t => t.trim());
                
                if (texts.length === 0) {
                    alert('Please enter some texts!');
                    return;
                }
                
                if (texts.length > 100) {
                    alert('Too many texts! Maximum 100 texts.');
                    return;
                }
                
                const resultDiv = document.getElementById('batch-result');
                resultDiv.innerHTML = `
                    <div class="loading">
                        <div class="spinner"></div>
                        <p>Processing ${texts.length} predictions...</p>
                    </div>
                `;
                
                try {
                    const response = await fetch('/predict/batch', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ texts: texts })
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    displayBatchResult(data);
                    
                } catch (error) {
                    resultDiv.innerHTML = `
                        <div class="prediction-item" style="border-color: #dc3545;">
                            <h3>Error</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                }
            }
            
            function displayBatchResult(data) {
                const resultDiv = document.getElementById('batch-result');
                let html = `
                    <div class="prediction-header">
                        <h3>Batch Results (${data.predictions.length} texts)</h3>
                        <span class="status healthy">Total: ${data.total_time_ms.toFixed(2)} ms | Avg: ${data.average_time_ms.toFixed(2)} ms</span>
                    </div>
                `;
                
                data.predictions.forEach((pred, index) => {
                    const shortText = pred.text.length > 50 ? pred.text.substring(0, 50) + '...' : pred.text;
                    html += `
                        <div class="prediction-item">
                            <div class="prediction-header">
                                <h4>Text ${index + 1}: "${shortText}"</h4>
                                <span>${pred.processing_time_ms.toFixed(2)} ms</span>
                            </div>
                            <p><strong>Top Prediction:</strong> ${pred.predicted_class.class_name} (ID: ${pred.predicted_class.class_id})</p>
                            <p><strong>Confidence:</strong> ${(pred.predicted_class.probability * 100).toFixed(2)}%</p>
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: ${pred.predicted_class.probability * 100}%"></div>
                            </div>
                        </div>
                    `;
                });
                
                resultDiv.innerHTML = html;
            }
            
            // Health check
            async function checkHealth() {
                const statusDiv = document.getElementById('health-status');
                
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    
                    statusDiv.innerHTML = `
                        <div class="prediction-item">
                            <div class="prediction-header">
                                <h3>API Health Status</h3>
                                <span class="status ${data.status === 'healthy' ? 'healthy' : 'error'}">${data.status}</span>
                            </div>
                            <p><strong>Model Loaded:</strong> ${data.model_loaded ? 'Yes' : 'No'}</p>
                            <p><strong>API Version:</strong> ${data.version}</p>
                            <p><strong>Timestamp:</strong> ${new Date(data.timestamp * 1000).toLocaleString()}</p>
                        </div>
                    `;
                } catch (error) {
                    statusDiv.innerHTML = `
                        <div class="prediction-item" style="border-color: #dc3545;">
                            <h3>API Unavailable</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                }
            }
            
            // Load system info
            async function loadSystemInfo() {
                const infoDiv = document.getElementById('system-info');
                
                try {
                    const response = await fetch('/info');
                    const data = await response.json();
                    
                    let originalMappingHtml = '';
                    if (data.original_mapping_keys && data.original_mapping_keys.length > 0) {
                        originalMappingHtml = `
                            <p><strong>Original Classes (first 5):</strong></p>
                            <div class="class-info">
                                ${data.original_mapping_keys.map(name => `<p>${name}</p>`).join('')}
                            </div>
                        `;
                    }
                    
                    infoDiv.innerHTML = `
                        <div class="prediction-item">
                            <div class="prediction-header">
                                <h3>Model Information</h3>
                            </div>
                            <p><strong>Model Name:</strong> ${data.model_name}</p>
                            <p><strong>Model Type:</strong> ${data.model_type}</p>
                            <p><strong>Number of Classes:</strong> ${data.num_labels}</p>
                            <p><strong>Max Text Length:</strong> ${data.max_length} characters</p>
                            <p><strong>Execution Device:</strong> ${data.device}</p>
                            <p><strong>Loaded At:</strong> ${new Date(data.loaded_at).toLocaleString()}</p>
                            ${originalMappingHtml}
                        </div>
                    `;
                } catch (error) {
                    infoDiv.innerHTML = `
                        <div class="prediction-item" style="border-color: #dc3545;">
                            <h3>Error Loading Info</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                }
            }
            
                        // Load classes list - простой вариант
            async function loadClassesList() {
                const classesDiv = document.getElementById('classes-list');
                
                try {
                    // Делаем тестовый запрос, чтобы получить все классы
                    const testResponse = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text: "test banking question" })
                    });
                    
                    if (!testResponse.ok) {
                        throw new Error(`Failed to load classes: ${testResponse.status}`);
                    }
                    
                    const testData = await testResponse.json();
                    
                    // Получаем информацию о модели для общего количества классов
                    const infoResponse = await fetch('/info');
                    const infoData = await infoResponse.ok ? await infoResponse.json() : { num_labels: testData.all_predictions.length };
                    
                    let html = `<p><strong>Total Classes:</strong> ${infoData.num_labels}</p>`;
                    
                    // Показываем все классы в отсортированном порядке
                    html += `<div style="max-height: 400px; overflow-y: auto; margin-top: 20px;">`;
                    
                    // Сортируем классы по ID (от 0 до N)
                    testData.all_predictions.sort((a, b) => a.class_id - b.class_id);
                    
                    testData.all_predictions.forEach(pred => {
                        html += `
                            <div class="class-info" style="margin-bottom: 10px;">
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <span style="font-weight: bold; color: #667eea;">ID ${pred.class_id}</span>
                                        <span style="margin-left: 10px;">${pred.class_name}</span>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                    
                    html += `</div>`;
                    
                    classesDiv.innerHTML = html;
                    
                } catch (error) {
                    classesDiv.innerHTML = `
                        <div class="prediction-item" style="border-color: #dc3545;">
                            <h3>Error Loading Classes</h3>
                            <p>${error.message}</p>
                        </div>
                    `;
                }
            }
            
            // Clear functions
            function clearSingle() {
                document.getElementById('single-text').value = '';
                document.getElementById('single-result').innerHTML = '';
            }
            
            function clearBatch() {
                document.getElementById('batch-texts').value = '';
                document.getElementById('batch-result').innerHTML = '';
            }
            
            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                checkHealth();
                loadSystemInfo();
            });
        </script>
    </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья API"""
    try:
        return HealthResponse(
            status="healthy",
            model_loaded=model_manager.model is not None,
            timestamp=time.time(),
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info", response_model=ModelInfoResponse)
async def model_info():
    """Получение информации о модели"""
    try:
        if model_manager.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        info = model_manager.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Предсказание для одного текста"""
    try:
        start_time = time.time()
        
        # Валидация текста
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if len(request.text) > MAX_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Text too long. Maximum {MAX_LENGTH} characters"
            )
        
        # Выполнение предсказания
        result = model_manager.predict_single(request.text)
        
        # Логирование с названием класса
        logger.info(
            f"Single prediction: '{request.text[:50]}...' -> "
            f"Class: {result['predicted_class']['class_name']} "
            f"(ID: {result['predicted_class']['class_id']}) "
            f"Confidence: {result['predicted_class']['confidence']:.2%} "
            f"Time: {result['processing_time_ms']:.2f}ms"
        )
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Пакетное предсказание"""
    try:
        # Валидация
        if not request.texts:
            raise HTTPException(status_code=400, detail="Texts list cannot be empty")
        
        if len(request.texts) > BATCH_SIZE_LIMIT:
            raise HTTPException(
                status_code=400,
                detail=f"Too many texts. Maximum {BATCH_SIZE_LIMIT} texts per request"
            )
        
        # Проверка длины каждого текста
        for i, text in enumerate(request.texts):
            if len(text) > MAX_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail=f"Text at position {i} is too long. Maximum {MAX_LENGTH} characters"
                )
        
        # Выполнение предсказания
        result = model_manager.predict_batch(request.texts)
        
        # Логирование с названиями классов
        logger.info(
            f"Batch prediction: {len(request.texts)} texts processed in "
            f"{result['total_time_ms']:.2f}ms (avg {result['average_time_ms']:.2f}ms per text)"
        )
        
        # Логируем топ-3 предсказания
        for i, pred in enumerate(result['predictions'][:3]):
            top_pred = pred['predicted_class']
            logger.info(
                f"  Text {i+1}: '{request.texts[i][:40]}...' -> "
                f"{top_pred['class_name']} ({top_pred['probability']:.2%})"
            )
        
        return BatchPredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Новый endpoint для получения списка всех классов
@app.get("/classes")
async def get_classes():
    """Получение списка всех классов с названиями"""
    try:
        if model_manager.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Создаем тестовый запрос, чтобы получить все классы
        test_text = "test"
        test_result = model_manager.predict_single(test_text)
        
        # Извлекаем все классы из результатов
        all_classes = []
        for pred in test_result['all_predictions']:
            all_classes.append({
                "class_id": pred["class_id"],
                "class_name": pred["class_name"]
            })
        
        return {
            "total_classes": len(all_classes),
            "classes": all_classes
        }
        
    except Exception as e:
        logger.error(f"Failed to get classes list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Обработчик ошибок HTTP"""
    error_response = ErrorResponse(
        error="HTTP Exception",
        detail=exc.detail,
        timestamp=time.time()
    )
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Обработчик общих исключений"""
    error_response = ErrorResponse(
        error="Internal Server Error",
        detail=str(exc),
        timestamp=time.time()
    )
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )

# ==================== ЗАПУСК ПРИЛОЖЕНИЯ ====================

def main():
    """Основная функция запуска"""
    try:
        # Вывод информации о конфигурации
        logger.info("=" * 60)
        logger.info("ALBERT Intent Classifier API")
        logger.info("=" * 60)
        logger.info(f"Model path: {MODEL_PATH}")
        logger.info(f"Label mapping file: {LABEL_MAPPING_FILE}")
        logger.info(f"API host: {API_HOST}")
        logger.info(f"API port: {API_PORT}")
        logger.info(f"Max text length: {MAX_LENGTH}")
        logger.info(f"Batch size limit: {BATCH_SIZE_LIMIT}")
        logger.info(f"Log level: {LOG_LEVEL}")
        logger.info(f"Log to file: {LOG_TO_FILE}")
        logger.info("=" * 60)
        
        # Проверка существования модели
        if not Path(MODEL_PATH).exists():
            logger.error(f"Model path does not exist: {MODEL_PATH}")
            logger.error("Please set the correct MODEL_PATH environment variable")
            sys.exit(1)
        
        # Проверка существования файла с маппингом классов
        if LABEL_MAPPING_FILE and Path(LABEL_MAPPING_FILE).exists():
            logger.info(f"Found label mapping file: {LABEL_MAPPING_FILE}")
        else:
            logger.warning(f"Label mapping file not found: {LABEL_MAPPING_FILE}")
            logger.warning("Will try to load class names from model directory")
        
        # Загрузка модели
        logger.info("Loading model...")
        model_manager.load_model()
        
        # Запуск сервера
        logger.info(f"Starting server on {API_HOST}:{API_PORT}")
        logger.info(f"API documentation: http://{API_HOST}:{API_PORT}/docs")
        logger.info(f"Web interface: http://{API_HOST}:{API_PORT}/")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        uvicorn.run(
            app,
            host=API_HOST,
            port=API_PORT,
            log_level="info" if not API_DEBUG else "debug",
            access_log=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down API server...")
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()