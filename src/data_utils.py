from typing import Tuple, List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RU_2CH_PIKABU_PATH, RU_OK_DATA_PATH

# теперь у нас один бинарный признак токсичности
TOXIC_LABELS: List[str] = ["toxic"]


def load_russian_toxic_data(
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Загружает и объединяет два русскоязычных датасета токсичности:
    - Russian Language Toxic Comments (2ch + Pikabu) [CSV]
    - Toxic Russian Comments from ok.ru [TXT, fastText-формат]

    Возвращает train/test разбиение:
    X_train, X_test, y_train, y_test
    """

    # 1. Russian Language Toxic Comments (2ch + Pikabu), Kaggle.
    # Обычно там столбцы: "comment" и "toxic".
    df1 = pd.read_csv(RU_2CH_PIKABU_PATH)

    # приводим к общей схеме: text, toxic
    if "text" not in df1.columns:
        if "comment" in df1.columns:
            df1 = df1.rename(columns={"comment": "text"})
    df1 = df1[["text", "toxic"]]
    df1["toxic"] = df1["toxic"].astype(int)

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
            if label_token == "__label__NORMAL":
                label = 0
            else:
                label = 1  # любое отклонение от нормы считаем токсичным

            rows.append({"text": text, "toxic": label})

    df2 = pd.DataFrame(rows)

    # 3. Объединяем
    df = pd.concat([df1, df2], ignore_index=True)
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str).str.strip()
    df["toxic"] = df["toxic"].astype(int)

    X = df["text"]
    # для OneVsRestClassifier удобно иметь (n_samples, 1)
    y = df["toxic"].values.reshape(-1, 1)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)