"""
FastAPI веб-сервис для прогнозирования дефолта по кредитным картам.

Эндпоинты:
- POST /predict - прогноз дефолта с A/B-тестированием
- GET /health - проверка работоспособности
- GET /ab-stats - статистика A/B-теста
"""

import logging
from datetime import datetime
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ABStatsResponse,
    CreditCardFeatures
)
from app.model_loader import get_model_loader
from app.ab_test import get_ab_router
from app import __version__

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Создание приложения
app = FastAPI(
    title="Credit Card Default Prediction API",
    description="ML-сервис для прогнозирования дефолта по кредитным картам с A/B-тестированием",
    version=__version__
)

# Загрузка моделей при старте
@app.on_event("startup")
async def startup_event():
    """Загрузка моделей при старте сервиса."""
    logger.info("Загрузка ML-моделей")
    try:
        loader = get_model_loader()
        loader.load_all()
        logger.info("Модели успешно загружены")
    except Exception as e:
        logger.error(f"Ошибка загрузки моделей: {e}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Корневой эндпоинт с информацией о сервисе."""
    return {
        "service": "Credit Card Default Prediction API",
        "version": __version__,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "ab-stats": "/ab-stats"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Проверка работоспособности сервиса.

    Возвращает статус сервиса и информацию о загруженных моделях.
    """
    loader = get_model_loader()
    return HealthResponse(
        status="ok",
        version=__version__,
        models_loaded=loader._model_v1 is not None
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest, http_request: Request):
    """
    Прогноз дефолта по кредитной карте.

    Если указан user_id, запросы распределяются между версиями моделей v1 и v2
    для A/B-тестирования (50/50 распределение).

    Args:
        request: Запрос с признаками клиента
        http_request: HTTP-запрос для логирования

    Returns:
        Прогноз дефолта с вероятностью и версией модели
    """
    request_id = str(uuid4())[:8]
    start_time = datetime.utcnow()

    try:
        # A/B-тестирование: выбор версии модели
        router = get_ab_router()
        model_version = router.get_model_version(request.user_id)

        # Прогноз
        loader = get_model_loader()
        features_dict = request.features.model_dump(by_alias=True)
        prediction, probability = loader.predict(features_dict, model_version)

        # Логирование для A/B-теста
        router.log_prediction(request.user_id, model_version, prediction, probability)

        # Логирование запроса (JSON)
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        log_entry = {
            "request_id": request_id,
            "timestamp": start_time.isoformat(),
            "duration_ms": round(duration_ms, 2),
            "user_id": request.user_id or "unknown",
            "model_version": model_version,
            "prediction": prediction,
            "probability": round(probability, 4)
        }
        logger.info(f"Prediction: {json_log(log_entry)}")

        return PredictionResponse(
            prediction=prediction,
            probability=round(probability, 4),
            model_version=model_version
        )

    except ValueError as e:
        logger.error(f"Validation error [{request_id}]: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error [{request_id}]: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/ab-stats", response_model=ABStatsResponse, tags=["A/B Testing"])
async def ab_stats():
    """
    Получить статистику A/B-теста.

    Возвращает количество прогнозов, сделанных каждой версией модели
    с момента запуска сервиса.
    """
    router = get_ab_router()
    stats = router.get_stats()
    return ABStatsResponse(**stats)


def json_log(obj: dict) -> str:
    """Преобразовать словарь в JSON для логирования."""
    import json
    return json.dumps(obj, ensure_ascii=False)


# Middleware для логирования всех запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Логирование всех HTTP-запросов."""
    start_time = datetime.utcnow()

    response = await call_next(request)

    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
    log_entry = {
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": round(duration_ms, 2)
    }
    logger.info(f"Request: {json_log(log_entry)}")

    return response


# Обработка ошибок
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Глобальный обработчик исключений."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
