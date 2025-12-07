from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import joblib
import numpy as np

from .config import BASELINE_MODEL_PATH
from .data_utils import TOXIC_LABELS


def _risk_level(conflict_score: float, threshold: float) -> str:
    if conflict_score >= max(threshold, 0.7):
        return "high"
    elif conflict_score >= 0.4:
        return "medium"
    else:
        return "low"


@dataclass
class PredictionResult:
    text: str
    conflict_score: float
    risk_level: str
    threshold: float
    labels: Dict[str, float]
    model: str = "ml_ru"  # подчёркиваем, что это русская ML-модель


class ConflictPredictionService:
    """
    Сервис инференса для русскоязычной baseline-модели токсичности.

    - загружает модель из models/baseline_model.pkl
    - даёт предсказание вероятности токсичности (toxic)
    """

    def __init__(
        self,
        model_path=BASELINE_MODEL_PATH,
        base_threshold: float = 0.7,
    ) -> None:
        self.model_path = model_path
        self.base_threshold = base_threshold
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        """
        Загружаем обученную baseline-модель, если файл существует.
        """
        try:
            print(f"[Service] Загружаем baseline-модель из {self.model_path} ...")
            self.model = joblib.load(self.model_path)
        except FileNotFoundError:
            print(f"[Service] ВНИМАНИЕ: файл модели {self.model_path} не найден. "
                  f"Сначала обучите модель командой: python -m src.baseline_model")
            self.model = None

    def is_ready(self) -> bool:
        """
        Проверка, загружена ли модель.
        """
        return self.model is not None

    def predict_single(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> PredictionResult:
        """
        Предсказание для одного текста.

        Возвращает:
        - вероятность токсичности (conflict_score)
        - уровень риска (risk_level)
        - словарь labels с одной меткой: {"toxic": p_toxic}
        """
        if self.model is None:
            raise RuntimeError("Модель не загружена. Сначала обучите её и перезапустите сервис.")

        # predict_proba вернёт массив формы (1, n_labels)
        probs: np.ndarray = self.model.predict_proba([text])[0]

        # у нас одна метка TOXIC_LABELS = ["toxic"], но оставим код универсальным
        labels_probs: Dict[str, float] = {
            TOXIC_LABELS[i]: float(probs[i]) for i in range(len(TOXIC_LABELS))
        }

        # конфликтный "скор" — максимальная вероятность токсичности
        conflict_score = float(probs.max())

        thr = threshold if threshold is not None else self.base_threshold
        risk_level = _risk_level(conflict_score, thr)

        return PredictionResult(
            text=text,
            conflict_score=conflict_score,
            risk_level=risk_level,
            threshold=thr,
            labels=labels_probs,
            model="ml_ru",
        )

    def predict_batch(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
    ) -> List[PredictionResult]:
        """
        Пакетный анализ списка сообщений.
        """
        if self.model is None:
            raise RuntimeError("Модель не загружена. Сначала обучите её и перезапустите сервис.")

        thr = threshold if threshold is not None else self.base_threshold

        probs_all: np.ndarray = self.model.predict_proba(texts)

        results: List[PredictionResult] = []
        for text, probs in zip(texts, probs_all):
            labels_probs: Dict[str, float] = {
                TOXIC_LABELS[i]: float(probs[i]) for i in range(len(TOXIC_LABELS))
            }
            conflict_score = float(probs.max())
            risk_level = _risk_level(conflict_score, thr)

            results.append(
                PredictionResult(
                    text=text,
                    conflict_score=conflict_score,
                    risk_level=risk_level,
                    threshold=thr,
                    labels=labels_probs,
                    model="ml_ru",
                )
            )

        return results

    def predict_single_as_dict(
        self,
        text: str,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Удобная обёртка для Flask: возвращаем dict, готовый к jsonify.
        """
        result = self.predict_single(text, threshold)
        return {
            "text": result.text,
            "model": result.model,
            "conflict_score": result.conflict_score,
            "risk_level": result.risk_level,
            "threshold": result.threshold,
            "labels": result.labels,
        }

    def predict_batch_as_dicts(
        self,
        texts: List[str],
        threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Аналогично, но сразу список dict'ов.
        """
        results = self.predict_batch(texts, threshold)
        return [
            {
                "text": r.text,
                "model": r.model,
                "conflict_score": r.conflict_score,
                "risk_level": r.risk_level,
                "threshold": r.threshold,
                "labels": r.labels,
            }
            for r in results
        ]