import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from .config import DATA_PATH, RANDOM_STATE

# Метки токсичности из датасета Jigsaw
TOXIC_LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Загружаем датасет Jigsaw из локального файла train.csv.
    Файл предварительно скачивается через браузер и
    сохраняется в папку data/ проекта.
    """
    df = pd.read_csv(path)
    cols = ["comment_text"] + TOXIC_LABELS
    df = df[cols].dropna(subset=["comment_text"])
    return df


def train_val_test_split(df: pd.DataFrame):
    """
    Разбиение данных на train/val/test для моделей машинного обучения.

    Для baseline-модели достаточно train/test, но оставим и валидацию,
    чтобы можно было при желании использовать и в других экспериментах.
    """
    X = df["comment_text"].astype(str).values
    y = df[TOXIC_LABELS].values

    # признак "есть хоть одна токсичная метка" используем для стратификации
    stratify_labels = (y.sum(axis=1) > 0)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    # Валидацию при желании можно не использовать в baseline,
    # но вернем её "на будущее"
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=(y_train_full.sum(axis=1) > 0),
    )

    return X_train, X_val, X_test, y_train, y_val, y_test