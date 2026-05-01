"""
Pydantic схемы для валидации запросов и ответов API.

Соответствуют структуре датасета UCI Credit Card Default.
"""

from pydantic import BaseModel, Field
from typing import Optional


class CreditCardFeatures(BaseModel):
    """
    Признаки клиента кредитной карты для прогнозирования дефолта.
    """

    limit_bal: float = Field(..., alias="LIMIT_BAL", description="Сумма кредитного лимита")
    sex: int = Field(..., ge=1, le=2, description="Пол (1=мужской, 2=женский)")
    education: int = Field(..., ge=1, le=6, description="Образование (1=graduate, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)")
    marriage: int = Field(..., ge=1, le=3, description="Семейное положение (1=married, 2=single, 3=others)")
    age: int = Field(..., ge=21, le=79, description="Возраст в годах")

    # История платежей (PAY_0 = PAY_1, статус погашения за сентябрь)
    pay_0: int = Field(..., alias="PAY_0", description="Статус погашения в сентябре")
    pay_2: int = Field(..., alias="PAY_2", description="Статус погашения в августе")
    pay_3: int = Field(..., alias="PAY_3", description="Статус погашения в июле")
    pay_4: int = Field(..., alias="PAY_4", description="Статус погашения в июне")
    pay_5: int = Field(..., alias="PAY_5", description="Статус погашения в мае")
    pay_6: int = Field(..., alias="PAY_6", description="Статус погашения в апреле")

    # Суммы счетов (выписка)
    bill_amt1: float = Field(..., alias="BILL_AMT1", description="Сумма счета в сентябре")
    bill_amt2: float = Field(..., alias="BILL_AMT2", description="Сумма счета в августе")
    bill_amt3: float = Field(..., alias="BILL_AMT3", description="Сумма счета в июле")
    bill_amt4: float = Field(..., alias="BILL_AMT4", description="Сумма счета в июне")
    bill_amt5: float = Field(..., alias="BILL_AMT5", description="Сумма счета в мае")
    bill_amt6: float = Field(..., alias="BILL_AMT6", description="Сумма счета в апреле")

    # Суммы платежей
    pay_amt1: float = Field(..., alias="PAY_AMT1", description="Сумма платежа в сентябре")
    pay_amt2: float = Field(..., alias="PAY_AMT2", description="Сумма платежа в августе")
    pay_amt3: float = Field(..., alias="PAY_AMT3", description="Сумма платежа в июле")
    pay_amt4: float = Field(..., alias="PAY_AMT4", description="Сумма платежа в июне")
    pay_amt5: float = Field(..., alias="PAY_AMT5", description="Сумма платежа в мае")
    pay_amt6: float = Field(..., alias="PAY_AMT6", description="Сумма платежа в апреле")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "LIMIT_BAL": 50000,
                "SEX": 2,
                "EDUCATION": 2,
                "MARRIAGE": 1,
                "AGE": 30,
                "PAY_0": 0,
                "PAY_2": 0,
                "PAY_3": 0,
                "PAY_4": 0,
                "PAY_5": 0,
                "PAY_6": 0,
                "BILL_AMT1": 1000,
                "BILL_AMT2": 2000,
                "BILL_AMT3": 1500,
                "BILL_AMT4": 1800,
                "BILL_AMT5": 1200,
                "BILL_AMT6": 900,
                "PAY_AMT1": 500,
                "PAY_AMT2": 1500,
                "PAY_AMT3": 1000,
                "PAY_AMT4": 1200,
                "PAY_AMT5": 800,
                "PAY_AMT6": 600
            }
        }


class PredictionRequest(BaseModel):
    """Запрос на прогноз дефолта."""

    user_id: Optional[str] = Field(None, description="ID пользователя для A/B-тестирования")
    features: CreditCardFeatures


class PredictionResponse(BaseModel):
    """Ответ с прогнозом дефолта."""

    prediction: int = Field(..., description="Прогноз (0=нет дефолта, 1=дефолт)")
    probability: float = Field(..., ge=0, le=1, description="Вероятность дефолта")
    model_version: str = Field(..., description="Версия модели (v1 или v2)")


class HealthResponse(BaseModel):
    """Ответ health check."""

    status: str
    version: str
    models_loaded: bool


class ABStatsResponse(BaseModel):
    """Ответ со статистикой A/B-теста."""

    v1_predictions: int
    v2_predictions: int
    total_predictions: int
