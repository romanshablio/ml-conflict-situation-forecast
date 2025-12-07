from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from .config import BASELINE_MODEL_PATH
from .data_utils import load_russian_toxic_data, TOXIC_LABELS
import joblib


def train_baseline_model():
    print("Загружаем данные (russian baseline)...")
    X_train, X_test, y_train, y_test = load_russian_toxic_data()

    print("Создаём pipeline TF-IDF + OneVsRest(LogisticRegression)...")
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=100_000,
                    ngram_range=(1, 2),
                    lowercase=True,
                ),
            ),
            (
                "clf",
                OneVsRestClassifier(
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1000,
                        n_jobs=12,
                    )
                ),
            ),
        ]
    )

    print("Обучаем baseline-модель...")
    pipeline.fit(X_train, y_train)

    print("Оцениваем качество на тестовой выборке...")
    y_pred = pipeline.predict(X_test)
    print(classification_report(
        y_test, y_pred,
        target_names=["non_toxic", "toxic"]
    ))

    BASELINE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Сохраняем baseline-модель в {BASELINE_MODEL_PATH} ...")
    joblib.dump(pipeline, BASELINE_MODEL_PATH)


if __name__ == "__main__":
    train_baseline_model()