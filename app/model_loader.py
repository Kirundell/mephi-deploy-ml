"""
Модуль загрузки ML-моделей для прогнозирования дефолта.

Поддерживает загрузку двух версий моделей для A/B-тестирования.
"""

import os
import joblib
from typing import Tuple, List
import numpy as np


# Пути к моделям (относительно директории models/)
_MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_MODEL_V1_PATH = os.path.join(_MODELS_DIR, 'model_v1.pkl')
_MODEL_V2_PATH = os.path.join(_MODELS_DIR, 'model_v2.pkl')
_SCALER_PATH = os.path.join(_MODELS_DIR, 'scaler.pkl')
_FEATURE_NAMES_PATH = os.path.join(_MODELS_DIR, 'feature_names.pkl')


class ModelLoader:
    """Класс для загрузки и использования ML-моделей."""

    def __init__(self):
        self._model_v1 = None
        self._model_v2 = None
        self._scaler = None
        self._feature_names = None

    def load_all(self) -> None:
        """Загрузить все модели и компоненты."""
        self._model_v1 = joblib.load(_MODEL_V1_PATH)
        self._model_v2 = joblib.load(_MODEL_V2_PATH)
        self._scaler = joblib.load(_SCALER_PATH)
        self._feature_names = joblib.load(_FEATURE_NAMES_PATH)

    @property
    def model_v1(self):
        """Получить модель v1 (контрольная)."""
        if self._model_v1 is None:
            self.load_all()
        return self._model_v1

    @property
    def model_v2(self):
        """Получить модель v2 (тестовая)."""
        if self._model_v2 is None:
            self.load_all()
        return self._model_v2

    @property
    def scaler(self):
        """Получить scaler для масштабирования."""
        if self._scaler is None:
            self.load_all()
        return self._scaler

    @property
    def feature_names(self) -> List[str]:
        """Получить список названий признаков."""
        if self._feature_names is None:
            self.load_all()
        return self._feature_names

    def predict(
        self,
        features: dict,
        model_version: str = "v1"
    ) -> Tuple[int, float]:
        """
        Сделать прогноз дефолта.

        Args:
            features: Словарь с признаками клиента
            model_version: Версия модели ("v1" или "v2")

        Returns:
            Кортеж (prediction, probability)
            - prediction: 0 или 1 (дефолт)
            - probability: вероятность дефолта
        """
        # Выбор модели
        model = self.model_v1 if model_version == "v1" else self.model_v2

        # Подготовка признаков
        feature_values = [features.get(name, 0) for name in self.feature_names]
        X = np.array([feature_values])

        # Масштабирование
        X_scaled = self.scaler.transform(X)

        # Прогноз
        prediction = int(model.predict(X_scaled)[0])
        probability = float(model.predict_proba(X_scaled)[0, 1])

        return prediction, probability

    def validate_features(self, features: dict) -> Tuple[bool, str]:
        """
        Валидация входных признаков.

        Args:
            features: Словарь с признаками

        Returns:
            Кортеж (is_valid, error_message)
        """
        if not features:
            return False, "Пустой запрос"

        missing = set(self.feature_names) - set(features.keys())
        if missing:
            return False, f"Missing features: {missing}"

        return True, ""


# Глобальный экземпляр загрузчика
_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """Получить глобальный экземпляр загрузчика моделей."""
    return _loader


def predict(features: dict, model_version: str = "v1") -> dict:
    """
    Прогноз дефолта.

    Args:
        features: Признаки клиента
        model_version: Версия модели

    Returns:
        Словарь с результатом прогноза
    """
    loader = get_model_loader()
    is_valid, error = loader.validate_features(features)

    if not is_valid:
        raise ValueError(error)

    prediction, probability = loader.predict(features, model_version)

    return {
        "prediction": prediction,
        "probability": probability,
        "model_version": model_version
    }
