from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional, Any

import joblib
import numpy as np

from .baseline_model import train_baseline_model
from .config import BASELINE_MODEL_PATH, RU_2CH_PIKABU_PATH, RU_OK_DATA_PATH
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
        auto_train: bool = True,
    ) -> None:
        self.model_path = Path(model_path)
        self.base_threshold = base_threshold
        self.auto_train = auto_train
        self.model = None

        self._ensure_model_loaded()

    def _datasets_available(self) -> bool:
        return RU_2CH_PIKABU_PATH.exists() and RU_OK_DATA_PATH.exists()

    def _load_model(self) -> bool:
        """
        Пытаемся загрузить обученную baseline-модель, если файл существует.
        """
        try:
            print(f"[Service] Загружаем baseline-модель из {self.model_path} ...")
            self.model = joblib.load(self.model_path)
            return True
        except FileNotFoundError:
            print(
                f"[Service] ВНИМАНИЕ: файл модели {self.model_path} не найден."
            )
            self.model = None
            return False
        except Exception as exc:  # noqa: BLE001
            print(f"[Service] Ошибка при загрузке модели: {exc}")
            self.model = None
            return False

    def _ensure_model_loaded(self) -> None:
        """
        Если модель отсутствует, по возможности обучаем её на локальных датасетах
        и только потом загружаем. Это позволяет сразу получать реальные предсказания
        без отдельного шага обучения.
        """
        if self._load_model():
            return

        if not self.auto_train:
            print("[Service] Автообучение отключено, модель не загружена.")
            return

        if not self._datasets_available():
            raise RuntimeError(
                "Файл модели отсутствует, а датасеты не найдены. "
                "Проверьте, что в каталоге data/ есть ru_toxic_2ch_pikabu.csv "
                "и ru_toxic_ok.txt или обучите модель вручную."
            )

        print("[Service] Автоматически обучаем baseline-модель на датасетах...")
        train_baseline_model()
        if not self._load_model():
            raise RuntimeError(
                "Не удалось загрузить baseline-модель после обучения. "
                "Проверьте логи обучения и состав датасетов."
            )

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
        Возвращает:
        - вероятность конфликтности (conflict_score)
        - уровень риска (risk_level)
        - словарь labels с вероятностями по типам токсичности:
          { "toxic": ..., "insult": ..., "threat": ..., "obscene": ... }
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
