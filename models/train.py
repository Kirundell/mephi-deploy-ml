"""
Скрипт обучения моделей для прогнозирования дефолта по кредитным картам.
Сохраняет две версии моделей для A/B-тестирования.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


def main():
    print("Обучение моделей прогнозирования дефолта")

    # Загрузка данных
    print("\n[1/7] Загрузка данных")
    data = pd.read_csv('UCI_Credit_Card.csv')
    print(f"Размер датасета: {data.shape}")
    print(f"Доля дефолта: {data['default.payment.next.month'].mean():.2%}")

    # Подготовка данных
    print("\n[2/7] Подготовка данных")
    X = data.drop(['ID', 'default.payment.next.month'], axis=1)
    y = data['default.payment.next.month']
    feature_names = list(X.columns)
    print(f"Признаков: {X.shape[1]}")

    # Разделение на train/test
    print("\n[3/7] Разделение на train/test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    # Масштабирование
    print("\n[4/7] Масштабирование признаков")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Данные масштабированы")

    # Обучение модели v1 (LogisticRegression)
    print("\n[5/7] Обучение моделей")
    print("Обучение model_v1 (LogisticRegression)")
    model_v1 = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced'
    )
    model_v1.fit(X_train_scaled, y_train)

    # Обучение модели v2 (GradientBoosting)
    print("Обучение model_v2 (GradientBoosting)")
    model_v2 = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_samples_split=5,
        random_state=42
    )
    model_v2.fit(X_train_scaled, y_train)

    # Оценка моделей
    print("\n[6/7] Оценка моделей")
    y_pred_v1 = model_v1.predict(X_test_scaled)
    y_proba_v1 = model_v1.predict_proba(X_test_scaled)[:, 1]
    y_pred_v2 = model_v2.predict(X_test_scaled)
    y_proba_v2 = model_v2.predict_proba(X_test_scaled)[:, 1]

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    v1_scores = [
        accuracy_score(y_test, y_pred_v1),
        precision_score(y_test, y_pred_v1),
        recall_score(y_test, y_pred_v1),
        f1_score(y_test, y_pred_v1),
        roc_auc_score(y_test, y_proba_v1)
    ]
    v2_scores = [
        accuracy_score(y_test, y_pred_v2),
        precision_score(y_test, y_pred_v2),
        recall_score(y_test, y_pred_v2),
        f1_score(y_test, y_pred_v2),
        roc_auc_score(y_test, y_proba_v2)
    ]

    comparison = pd.DataFrame({
        'Metric': metrics,
        'v1': [f"{s:.4f}" for s in v1_scores],
        'v2': [f"{s:.4f}" for s in v2_scores],
        'Diff': [f"{v2-v1:+.4f}" for v2, v1 in zip(v2_scores, v1_scores)]
    })
    print(comparison.to_string(index=False))

    # Сохранение моделей
    print("\n[7/7] Сохранение моделей")
    joblib.dump(model_v1, 'model_v1.pkl')
    print("Сохранено model_v1.pkl")
    joblib.dump(model_v2, 'model_v2.pkl')
    print("Сохранено model_v2.pkl")
    joblib.dump(scaler, 'scaler.pkl')
    print("Сохранено scaler.pkl")
    joblib.dump(feature_names, 'feature_names.pkl')
    print("Сохранено feature_names.pkl")


if __name__ == "__main__":
    main()
