"""
Модуль A/B-тестирования моделей.

Реализует роутер 50/50 для распределения запросов между версиями моделей.
Использует детерминированный хеш для стабильного распределения.
"""

import hashlib
import json
import os
from datetime import datetime
from typing import Optional, Tuple
from collections import defaultdict

# Путь к файлу с логами A/B-теста
_AB_LOG_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'ab_predictions.jsonl')


class ABTestRouter:
    """Роутер для A/B-тестирования моделей."""

    def __init__(self):
        self._stats = defaultdict(int)

    def get_model_version(self, user_id: str) -> str:
        """
        Определить версию модели для пользователя.

        Использует хеш user_id для детерминированного распределения 50/50.

        Args:
            user_id: Идентификатор пользователя

        Returns:
            "v1" или "v2"
        """
        if not user_id:
            # Без user_id используем v1 по умолчанию
            return "v1"

        # Хеш для детерминированного распределения
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        return "v1" if hash_value % 2 == 0 else "v2"

    def log_prediction(
        self,
        user_id: Optional[str],
        model_version: str,
        prediction: int,
        probability: float
    ) -> None:
        """
        Записать прогноз в лог для анализа A/B-теста.

        Args:
            user_id: ID пользователя
            model_version: Версия модели
            prediction: Прогноз (0 или 1)
            probability: Вероятность дефолта
        """
        self._stats[model_version] += 1

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id or "unknown",
            "model_version": model_version,
            "prediction": prediction,
            "probability": probability
        }

        try:
            with open(_AB_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Error logging prediction: {e}")

    def get_stats(self) -> dict:
        """Получить статистику распределения запросов."""
        total = sum(self._stats.values())
        return {
            "v1_predictions": self._stats.get("v1", 0),
            "v2_predictions": self._stats.get("v2", 0),
            "total_predictions": total
        }


# Глобальный экземпляр роутера
_router = ABTestRouter()


def get_ab_router() -> ABTestRouter:
    """Получить глобальный экземпляр A/B-роутера."""
    return _router
