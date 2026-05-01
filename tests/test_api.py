"""
Тесты для FastAPI ML-сервиса.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.ab_test import get_ab_router, ABTestRouter


# Фикстура для тестового клиента
@pytest.fixture
def client():
    """Создать тестовый HTTP-клиент."""
    return TestClient(app)


# Фикстура для сброса статистики A/B-теста
@pytest.fixture
def clean_ab_stats():
    """Сбросить статистику A/B-теста перед тестом."""
    router = get_ab_router()
    router._stats.clear()
    yield
    router._stats.clear()


# Тестовые данные
SAMPLE_FEATURES = {
    "limit_bal": 50000,
    "sex": 2,
    "education": 2,
    "marriage": 1,
    "age": 30,
    "pay_0": 0,
    "pay_2": 0,
    "pay_3": 0,
    "pay_4": 0,
    "pay_5": 0,
    "pay_6": 0,
    "bill_amt1": 1000,
    "bill_amt2": 2000,
    "bill_amt3": 1500,
    "bill_amt4": 1800,
    "bill_amt5": 1200,
    "bill_amt6": 900,
    "pay_amt1": 500,
    "pay_amt2": 1500,
    "pay_amt3": 1000,
    "pay_amt4": 1200,
    "pay_amt5": 800,
    "pay_amt6": 600
}


class TestHealthEndpoint:
    """Тесты для эндпоинта /health."""

    def test_health_returns_ok(self, client):
        """Проверяет, что health check возвращает статус ok."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert data["models_loaded"] is True


class TestPredictEndpoint:
    """Тесты для эндпоинта /predict."""

    def test_predict_returns_valid_response(self, client, clean_ab_stats):
        """Проверяет, что predict возвращает корректный ответ."""
        request_data = {
            "user_id": "test_user_1",
            "features": SAMPLE_FEATURES
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "model_version" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1
        assert data["model_version"] in ["v1", "v2"]

    def test_predict_without_user_id(self, client, clean_ab_stats):
        """Проверяет, что без user_id используется модель v1."""
        request_data = {
            "features": SAMPLE_FEATURES
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 200

        data = response.json()
        assert data["model_version"] == "v1"

    def test_predict_ab_routing(self, client, clean_ab_stats):
        """Проверяет, что A/B-роутер распределяет запросы."""
        # Делаем несколько запросов с разными user_id
        for i in range(10):
            request_data = {
                "user_id": f"user_{i}",
                "features": SAMPLE_FEATURES
            }
            client.post("/predict", json=request_data)

        # Проверяем статистику
        stats_response = client.get("/ab-stats")
        stats = stats_response.json()

        # Оба распределения должны иметь запросы
        assert stats["v1_predictions"] > 0
        assert stats["v2_predictions"] > 0
        assert stats["total_predictions"] == 10

    def test_predict_invalid_features(self, client):
        """Проверяет валидацию при некорректных признаках."""
        request_data = {
            "user_id": "test_user",
            "features": {
                "limit_bal": "invalid",
                "sex": 2,
                "education": 2,
                "marriage": 1,
                "age": 30
            }
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_predict_missing_features(self, client):
        """Проверяет валидацию при пропущенных признаках."""
        request_data = {
            "user_id": "test_user",
            "features": {
                "limit_bal": 50000,
                "sex": 2
            }
        }

        response = client.post("/predict", json=request_data)
        assert response.status_code == 422


class TestABStatsEndpoint:
    """Тесты для эндпоинта /ab-stats."""

    def test_ab_stats_returns_zero_initially(self, client, clean_ab_stats):
        """Проверяет, что изначально статистика пустая."""
        response = client.get("/ab-stats")
        assert response.status_code == 200

        data = response.json()
        assert data["v1_predictions"] == 0
        assert data["v2_predictions"] == 0
        assert data["total_predictions"] == 0

    def test_ab_stats_increments(self, client, clean_ab_stats):
        """Проверяет, что статистика обновляется."""
        # Делаем прогноз
        client.post("/predict", json={
            "user_id": "test_user",
            "features": SAMPLE_FEATURES
        })

        # Проверяем статистику
        response = client.get("/ab-stats")
        data = response.json()
        assert data["total_predictions"] == 1


class TestRootEndpoint:
    """Тесты для корневого эндпоинта."""

    def test_root_returns_service_info(self, client):
        """Проверяет, что корневой эндпоинт возвращает информацию о сервисе."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "service" in data
        assert "version" in data
        assert "endpoints" in data


class TestABRouter:
    """Юнит-тесты для A/B-роутера."""

    def test_router_returns_v1_or_v2(self):
        """Проверяет, что роутер возвращает только v1 или v2."""
        router = ABTestRouter()

        for i in range(100):
            user_id = f"user_{i}"
            version = router.get_model_version(user_id)
            assert version in ["v1", "v2"]

    def test_router_is_deterministic(self):
        """Проверяет, что роутер детерминирован для одного user_id."""
        router = ABTestRouter()
        user_id = "test_user_123"

        version1 = router.get_model_version(user_id)
        version2 = router.get_model_version(user_id)

        assert version1 == version2

    def test_router_distributes_evenly(self):
        """Проверяет примерно равномерное распределение."""
        router = ABTestRouter()
        versions = []

        for i in range(1000):
            user_id = f"user_{i}"
            version = router.get_model_version(user_id)
            versions.append(version)

        v1_count = versions.count("v1")
        v2_count = versions.count("v2")

        assert 400 < v1_count < 600
        assert 400 < v2_count < 600
