from typing import List, Dict

import numpy as np
import joblib

from .config import BASELINE_MODEL_PATH
from .data_utils import TOXIC_LABELS


class ConflictPredictionService:
    """
    Упрощённый сервис прогнозирования конфликтных ситуаций,
    использующий только baseline ML-модель (TF-IDF + LogisticRegression).

    Никакого TensorFlow здесь нет, всё работает на scikit-learn.
    """

    def __init__(self, baseline_model_path: str = BASELINE_MODEL_PATH):
        self.labels = TOXIC_LABELS
        self.baseline_model_path = baseline_model_path
        self.baseline_model = None
        self._load_baseline_model()

    def _load_baseline_model(self):
        try:
            print(f"[Service] Загружаем baseline-модель из {self.baseline_model_path} ...")
            self.baseline_model = joblib.load(self.baseline_model_path)
        except Exception as e:
            print(f"[Service] Не удалось загрузить baseline-модель: {e}")
            self.baseline_model = None

    def predict_single(self, text: str) -> Dict:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        if self.baseline_model is None:
            raise RuntimeError("Baseline-модель не загружена. Сначала обучите её через src.baseline_model.")

        # predict_proba у OneVsRestClassifier возвращает список массивов по каждому классу
        probs = self.baseline_model.predict_proba(texts)
        if isinstance(probs, list):
            probs = np.stack([p[:, 1] for p in probs], axis=1)

        results = []
        for i, text in enumerate(texts):
            labels_probs = {
                label: float(probs[i, j])
                for j, label in enumerate(self.labels)
            }
            conflict_score = float(np.max(probs[i]))  # максимальная вероятность по меткам
            results.append(
                {
                    "text": text,
                    "labels": labels_probs,
                    "conflict_score": conflict_score,
                    "model": "ml",
                }
            )
        return results