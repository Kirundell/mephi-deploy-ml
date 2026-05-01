# A/B-тестирование ML-моделей

## Обзор

Сервис поддерживает A/B-тестирование двух версий модели прогнозирования дефолта для сравнения их качества в реальных условиях.

## Гипотеза

**H0:** Модель v2 показывает тот же F1-score, что и модель v1  
**H1:** Модель v2 показывает лучший F1-score, чем модель v1

## Модели

| Версия | Модель | Описание | Параметры |
|--------|--------|----------|-----------|
| **v1** | LogisticRegression | Простая, интерпретируемая | max_iter=1000, class_weight='balanced' |
| **v2** | GradientBoostingClassifier | Сложная, высокая точность | n_estimators=200, max_depth=5, lr=0.05 |

**Почему эти модели?**

- LogisticRegression - линейная модель, высокий Recall (меньше пропущенных дефолтов)
- GradientBoosting - нелинейная модель, высокий Precision (меньше ложных срабатываний)

## Дизайн теста

### Распределение трафика

- **Метод:** Детерминированный хеш MD5 от user_id
- **Пропорция:** 50% v1, 50% v2
- **Стабильность:** Один пользователь всегда попадает в одну версию

```python
def get_model_version(user_id: str) -> str:
    hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    return "v1" if hash_value % 2 == 0 else "v2"
```

### Длительность

- **Минимум:** 2 недели
- **Или до:** достижения статистической значимости (power > 0.8)
- **Размер выборки:** минимум 1000 наблюдений на группу

## Метрики

### Технические метрики

| Метрика | Описание |
|---------|----------|
| **F1-score** | Основная метрика качества |
| **Recall** | Полнота обнаружения дефолтов |
| **Precision** | Точность прогнозов |
| **ROC-AUC** | Общая ранжирующая способность |

**Почему Recall важен?**

В кредитном скоринге ложноположительные (FP) и ложноотрицательные (FN) прогнозы имеют разную стоимость:

- **FN (пропустить дефолт):** банк теряет всю сумму кредита
- **FP (ложный дефолт):** банк теряет проценты от невыданного кредита

поэтому Recall важнее Precision

### Бизнес-метрики

#### 1. Expected Loss (Ожидаемые потери)

```
Expected Loss = FN_count * cost_FN + FP_count * cost_FP

где:
cost_FN = средняя сумма непогашенного кредита
cost_FP = упущенная выгода от процента по кредиту
```

#### 2. Approval Rate (Доля одобренных заявок)

```
Approval Rate = (TP + TN) / Total

при фиксированном уровне риска (threshold)
```

Более высокая доля одобрений при том же уровне риска = больше прибыли.

## Статистический анализ

### Bootstrap для F1-score

Поскольку F1-score имеет не нормальное распределение, используем бутстрап:

```python
def bootstrap_f1_score(y_true, y_pred, n_bootstrap=10000):
    scores = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        score = f1_score(y_true[indices], y_pred[indices])
        scores.append(score)
    return np.percentile(scores, [2.5, 97.5])  # 95% CI
```

### Критерий успешности

1. **F1_v2 > F1_v1** (point estimate)
2. **95% CI не пересекаются**
3. **p-value < 0.05**

### Альтернативные тесты

- **McNemar's test:** для сравнения двух классификаторов на одних данных
- **Permutation test:** непараметрический тест
- **Z-test for proportions:** для сравнения долей

## Логирование

Все прогнозы записываются в `models/ab_predictions.jsonl`:

```json
{"timestamp": "2026-05-01T18:00:00", "user_id": "customer_123", "model_version": "v2", "prediction": 0, "probability": 0.0776}
{"timestamp": "2026-05-01T18:00:01", "user_id": "customer_456", "model_version": "v1", "prediction": 1, "probability": 0.6523}
```

### Анализ логов

```python
import pandas as pd

# Чтение логов
df = pd.read_json('models/ab_predictions.jsonl', lines=True)

# Распределение по версиям
print(df['model_version'].value_counts())
```

## Мониторинг во время теста

1. **Sample Ratio Mismatch (SRM):** проверка, что распределение действительно 50/50
2. **Проверка на выбросы:** нет ли аномалий в данных
3. **AA-test:** предварительный тест, что v1 и v1 дают одинаковые результаты

## Принятие решения

### Если v2 выигрывает

- Внедряем v2 как основную модель
- Gradual rollout: 25% → 50% → 100%

### Если v2 проигрывает

- Анализ причин
- Возврат к v1
- Итерация над моделями

### Если нет статистической значимости

- Увеличить размер выборки
- Продлить тест
- Признать, что разница незначительна

## Пример команды для анализа

```bash
# Получить статистику
curl http://localhost:8000/ab-stats

# Скачать логи
scp user@server:/app/models/ab_predictions.jsonl ./

# Анализ в Python
python -c "
import pandas as pd
df = pd.read_json('ab_predictions.jsonl', lines=True)
print(df.groupby('model_version').agg({
    'prediction': ['count', 'mean'],
    'probability': 'mean'
}))
"
```

## Дальнейшее улучшение

1. **Multi-armed bandit:** динамическое перераспределение трафика
2. **CUPED:** уменьшение дисперсии с использованием ковариат
