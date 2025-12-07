import os
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from .config import BASELINE_MODEL_PATH, RANDOM_STATE
from .data_utils import load_raw_data, TOXIC_LABELS


def train_baseline_model():
    os.makedirs(os.path.dirname(BASELINE_MODEL_PATH), exist_ok=True)

    print("Загружаем данные (baseline)...")
    df = load_raw_data()
    X = df["comment_text"].astype(str).values
    y = df[TOXIC_LABELS].values

    # Для baseline делим просто train/test (валидацию можно сделать через cross-val при желании)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=(y.sum(axis=1) > 0),
    )

    print("Создаём pipeline TF-IDF + OneVsRest(LogisticRegression)...")
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=50000,
                    ngram_range=(1, 2),
                    stop_words=None  # при необходимости можно добавить список
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        max_iter=1000,
                        n_jobs=-1,
                        C=4.0,
                        solver="liblinear"
                    )
                ),
            ),
        ]
    )

    print("Обучаем baseline-модель...")
    pipeline.fit(X_train, y_train)

    print("Оцениваем качество на тестовой выборке...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=TOXIC_LABELS))

    print(f"Сохраняем baseline-модель в {BASELINE_MODEL_PATH} ...")
    joblib.dump(pipeline, BASELINE_MODEL_PATH)

    return pipeline


if __name__ == "__main__":
    train_baseline_model()