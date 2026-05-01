# ML-сервис прогнозирования дефолта по кредитным картам

Production-like веб-сервис машинного обучения для прогнозирования дефолта по кредитным картам с A/B-тестированием моделей.

## Описание проекта

Сервис реализует полный цикл внедрения ML-модели:
- Обучение и сохранение двух версий моделей (LogReg и GradientBoostingClassifier)
- FastAPI веб-сервис с REST API
- A/B-тестирование с роутером 50/50
- Docker-контейнеризация
- Документация

## Домен

**Кредитный скоринг**

Датасет: [UCI Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

## Структура проекта

```
course_task/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI приложение
│   ├── schemas.py           # Pydantic модели
│   ├── model_loader.py      # Загрузка моделей
│   └── ab_test.py           # A/B-тест роутер
├── models/
│   ├── UCI_Credit_Card.csv  # Датасет
│   ├── train.py             # Скрипт обучения моделей
│   ├── model_v1.pkl         # LogisticRegression
│   ├── model_v2.pkl         # GradientBoostingClassifier
│   ├── scaler.pkl           # StandardScaler
│   └── feature_names.pkl    # Названия признаков
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── tests/
│   └── test_api.py          # Тесты API
├── docs/
│   ├── ARCHITECTURE.md      # Архитектура
│   └── AB_TESTING.md        # A/B-тестирование
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

## Установка и запуск

### Локальный запуск

1. **Клонирование и установка зависимостей**

```bash
git clone <repository-url>
cd course_task
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. **Обучение моделей**

```bash
cd models
python train.py
```

3. **Запуск сервиса**

```bash
cd ..
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Сервис будет доступен по адресу: http://localhost:8000

### Запуск в Docker

1. **Сборка образа**

```bash
docker build -f docker/Dockerfile -t credit-default-api:latest .
```

2. **Запуск контейнера**

```bash
docker run -p 8000:8000 credit-default-api:latest
```

3. **Запуск через Docker Compose**

```bash
cd docker
docker-compose up
```

## API Эндпоинты

### POST /predict

Прогноз дефолта по кредитной карте.

**Запрос:**

```json
{
  "user_id": "customer_123",
  "features": {
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
}
```

**Ответ:**

```json
{
  "prediction": 0,
  "probability": 0.0776,
  "model_version": "v2"
}
```

### GET /health

Проверка работоспособности сервиса.

**Ответ:**

```json
{
  "status": "ok",
  "version": "1.0.0",
  "models_loaded": true
}
```

### GET /ab-stats

Статистика A/B-теста.

**Ответ:**

```json
{
  "v1_predictions": 45,
  "v2_predictions": 55,
  "total_predictions": 100
}
```

## Примеры запросов

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Прогноз
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test_user",
    "features": {
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
  }'

# Статистика A/B-теста
curl http://localhost:8000/ab-stats
```

### Python

```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "user_id": "customer_123",
    "features": {
        "limit_bal": 50000,
        "sex": 2,
        "education": 2,
        "marriage": 1,
        "age": 30,
        # ... остальные признаки
    }
}

response = requests.post(url, json=payload)
print(response.json())
```

## Тестирование

```bash
pytest tests/test_api.py -v
```

## A/B-тестирование

Сервис поддерживает A/B-тестирование двух версий моделей:
- **v1**: LogisticRegression
- **v2**: GradientBoostingClassifier

Распределение трафика: 50/50 на основе детерминированного хеша user_id.

Подробнее: [docs/AB_TESTING.md](docs/AB_TESTING.md)

## Метрики моделей

| Метрика | v1 (LogReg) | v2 (GradientBoosting) |
|---------|-------------|----------------------|
| Accuracy | 0.6797 | 0.8163 |
| Precision | 0.3672 | 0.6569 |
| Recall | **0.6202** | 0.3549 |
| F1 | 0.4613 | 0.4609 |
| ROC-AUC | 0.7081 | **0.7780** |

## Архитектура

Подробнее об архитектуре сервиса: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## Docker Hub

Образ доступен по адресу: `docker pull kirundell/credit-default-api:v1.0`

Ссылка: [https://hub.docker.com/r/kirundell/credit-default-api](https://hub.docker.com/r/kirundell/credit-default-api)

## Требования

- Python 3.11+
- scikit-learn 1.8.0+
- FastAPI 0.136+
- Docker
