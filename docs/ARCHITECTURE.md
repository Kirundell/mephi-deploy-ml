# Архитектура ML-сервиса

## Обзор

Сервис реализован как **монолитное приложение** на FastAPI, что оптимально для учебного проекта и MVP.

## Монолит vs Микросервисы

### Почему монолит?

Для данного учебного проекта выбран **монолитный подход** по следующим причинам:

1. **Простота разработки**: Один сервис проще разрабатывать, тестировать и отлаживать
2. **Меньше overhead**: Отсутствует необходимость в сервисной_mesh, API Gateway
3. **Быстрый запуск**: Нет задержек на межсервисные вызовы
4. **Единое развертывание**: Один Docker-образ вместо оркестрации нескольких

### Когда переходить к микросервисам?

Микросервисная архитектура могла бы быть полезна при:

- **Независимые команды разработки**: разные команды работают над разными компонентами
- **Различные требования к масштабированию**: модель требует GPU, API — нет
- **Разные языки/стеки**: preprocessing на Python, model serving на C++
- **Частые деплои**: изменения в одной части не требуют перезапуска всей системы

### Концепт микросервисной архитектуры

```
                    ┌─────────────┐
                    │  API Gateway│
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐      ┌──────────┐      ┌────────────┐
   │ Feature │      │  Model   │      │ Monitoring │
   │ Service │      │  Service │      │  Service   │
   └─────────┘      └──────────┘      └────────────┘
        │                  │
        └────────┬─────────┘
                 ▼
         ┌───────────────┐
         │  Feature Store│
         └───────────────┘
```

## Текущая архитектура

```
┌─────────────────────────────────────────────────────────┐
│                     FastAPI App                         │
│  ┌───────────┐  ┌───────────┐  ┌──────────────────┐   │
│  │  Routes   │  │  Schemas  │  │   A/B Router     │   │
│  └─────┬─────┘  └─────┬─────┘  └────────┬─────────┘   │
│        │               │                   │             │
│        └───────────────┼───────────────────┘             │
│                        ▼                                 │
│              ┌─────────────────────┐                    │
│              │   Model Loader      │                    │
│              └─────────┬───────────┘                    │
│                        ▼                                 │
└────────────────────────┼─────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐    ┌─────────┐    ┌──────────┐
   │ model_v1│    │ model_v2│    │  scaler  │
   └─────────┘    └─────────┘    └──────────┘
```

## Компоненты

### 1. API Layer (`app/main.py`)
- FastAPI приложение
- Эндпоинты: `/predict`, `/health`, `/ab-stats`
- Middleware для логирования
- Обработка ошибок

### 2. Data Validation (`app/schemas.py`)
- Pydantic модели для валидации запросов
- 23 признака кредитной карты
- Примеры для документации

### 3. Model Loader (`app/model_loader.py`)
- Загрузка моделей (joblib)
- Кеширование в памяти
- Масштабирование признаков
- Валидация входных данных

### 4. A/B Router (`app/ab_test.py`)
- Детерминированный роутер 50/50
- Логирование в JSONL
- Статистика распределения

### Модели для A/B-теста

| Версия | Модель | Параметры | Особенности |
|--------|--------|-----------|-------------|
| **v1** | LogisticRegression | max_iter=1000, class_weight='balanced' | Высокий Recall, интерпретируемая |
| **v2** | GradientBoostingClassifier | n_estimators=200, max_depth=5 | Высокий Precision, точнее |

## Логирование

### Формат логов

Все запросы логируются в JSON-формате:

```json
{
  "timestamp": "2026-05-01T18:00:00",
  "request_id": "abc12345",
  "method": "POST",
  "path": "/predict",
  "duration_ms": 15.23,
  "user_id": "customer_123",
  "model_version": "v2",
  "prediction": 0,
  "probability": 0.0776
}
```

### Мониторинг в Production

Для production-системы рекомендуется:

1. **ELK Stack** (Elasticsearch, Logstash, Kibana)
   - Сбор логов с нескольких экземпляров
   - Агрегация и поиск
   - Дашборды для мониторинга

2. **Prometheus + Grafana**
   - Метрики: latency, throughput, error rate
   - Алертинг при аномалиях
   - Визуализация распределения A/B-теста

3. **Jaeger / Zipkin**
   - Distributed tracing
   - Отслеживание запросов между сервисами

## Масштабирование

### RabbitMQ для асинхронной обработки

При масштабировании брокер очередей может использоваться для:

1. **Batch предсказания**
   - Накопление запросов в очереди
   - Обработка пачками для эффективности
   - Сглаживание пиковых нагрузок

2. **Асинхронное логирование**
   - Отправка логов в очередь
   - Отдельный сервис для записи

3. **Реализация концепта**

```
┌─────────┐    ┌──────────┐    ┌──────────┐
│   API   │───▶│ RabbitMQ │───▶│  Worker  │
└─────────┘    └──────────┘    └──────────┘
                     │
                     ▼
              ┌──────────┐
              │ Log Store│
              └──────────┘
```

**Пример кода (концепт):**

```python
# Producer (API)
import pika
connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()
channel.queue_declare(queue='predictions')

channel.basic_publish(
    exchange='',
    routing_key='predictions',
    body=json.dumps(prediction_request)
)

# Consumer (Worker)
def callback(ch, method, properties, body):
    request = json.loads(body)
    result = model.predict(request['features'])
    log_result(request['user_id'], result)

channel.basic_consume(queue='predictions', on_message_callback=callback)
channel.start_consuming()
```

## uWSGI + NGINX в Production

### uWSGI
- Application server для Python
- Управление процессами/воркерами
- Автоматический перезапуск при падениях
- Graceful reload

### NGINX
- Reverse proxy
- SSL termination
- Статические файлы
- Rate limiting
- Load balancing

**Пример конфигурации:**

```nginx
upstream ml_service {
    server 127.0.0.1:8001;
    server 127.0.0.1:8002;
    server 127.0.0.1:8003;
}

server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location /predict {
        limit_req zone=one burst=20 nodelay;
        proxy_pass http://ml_service;
    }
}
```

## ML Tools Overview

### DVC (Data Version Control)
- Контроль версий больших файлов данных
- Трекинг пайплайнов
- Воспроизводимость экспериментов

```bash
dvc init
dvc add data/UCI_Credit_Card.csv
dvc push
```

### MLflow
- Отслеживание экспериментов
- Хранение моделей (Model Registry)
- Сравнение метрик между запусками

```python
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "model")
```

## Безопасность

1. **Валидация входных данных** (Pydantic)
2. **Rate limiting** (NGINX)
3. **HTTPS** (SSL termination)
4. **Аутентификация** (JWT/OAuth2)
5. **Санитизация логов** (без PII данных)

## Резюме

Текущая монолитная архитектура оптимальна для:
- Учебных проектов
- MVP и PoC
- Небольших команд
- Простых use cases

Микросервисы имеют смысл при:
- Масштабировании команды
- Различных SLA для компонентов
- Независимых деплоях
- Разнородном стеке технологий
