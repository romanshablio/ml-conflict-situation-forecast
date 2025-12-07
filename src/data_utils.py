from typing import Tuple, List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RU_2CH_PIKABU_PATH, RU_OK_DATA_PATH

# четыре признака токсичности
TOXIC_LABELS: List[str] = ["toxic", "insult", "threat", "obscene"]


def load_russian_toxic_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Загружает и объединяет два русскоязычных датасета токсичности:
    - Russian Language Toxic Comments (2ch + Pikabu) [CSV]
    - Toxic Russian Comments from ok.ru [TXT, fastText-формат]

    Формирует мультилейбл-разметку по четырём признакам:
    - toxic   — общий факт токсичности (1, если есть любой тип токсичности)
    - insult  — оскорбления
    - threat  — угрозы
    - obscene — ненормативная / обсценная лексика

    Возвращает train/test разбиение: X_train, X_test, y_train, y_test,
    где y_* — DataFrame с колонками TOXIC_LABELS.
    """

    # 1. Russian Language Toxic Comments (2ch + Pikabu), Kaggle.
    # Обычно там столбцы: "comment" и "toxic" (0/1).
    df1 = pd.read_csv(RU_2CH_PIKABU_PATH)

    # приводим к общей схеме: text + 4 бинарных признака
    if "text" not in df1.columns:
        if "comment" in df1.columns:
            df1 = df1.rename(columns={"comment": "text"})

    df1 = df1[["text", "toxic"]].copy()
    df1["toxic"] = df1["toxic"].astype(int)

    # для 2ch+Pikabu знаем только факт токсичности:
    # если toxic == 1, считаем, что был "generic toxic" без детализации
    df1_labels = pd.DataFrame(
        {
            "toxic": df1["toxic"],
            "insult": 0,
            "threat": 0,
            "obscene": 0,
        }
    )

    df1 = pd.concat([df1[["text"]], df1_labels], axis=1)

    # 2. Toxic Russian Comments from ok.ru (fastText-формат)
    rows: List[Dict[str, object]] = []
    with open(RU_OK_DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                label_token, text = line.split(" ", 1)
            except ValueError:
                # строка без текста/метки — пропускаем
                continue

            # fastText-метки: __label__NORMAL, __label__INSULT, __label__THREAT, __label__OBSCENITY
            toxic = 0
            insult = 0
            threat = 0
            obscene = 0

            if label_token == "__label__NORMAL":
                pass  # все нули
            elif label_token == "__label__INSULT":
                toxic = 1
                insult = 1
            elif label_token == "__label__THREAT":
                toxic = 1
                threat = 1
            elif label_token == "__label__OBSCENITY":
                toxic = 1
                obscene = 1
            else:
                # неизвестная метка — пропускаем
                continue

            rows.append(
                {
                    "text": text,
                    "toxic": toxic,
                    "insult": insult,
                    "threat": threat,
                    "obscene": obscene,
                }
            )

    df2 = pd.DataFrame(rows)

    # 3. Объединяем оба корпуса
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()

    # Убедимся, что все метки целочисленные 0/1
    for col in TOXIC_LABELS:
        df[col] = df[col].fillna(0).astype(int)

    X = df["text"]
    y = df[TOXIC_LABELS].copy()

    return train_test_split(X, y, test_size=test_size, random_state=random_state)