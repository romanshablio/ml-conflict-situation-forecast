"""
Простой CLI для анализа текста через обученную baseline-модель.

Примеры:
  python -m src.cli --text "Пример сообщения для проверки"
  python -m src.cli --file samples.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from .model_service import ConflictPredictionService


def _load_texts(args: argparse.Namespace) -> List[str]:
    if args.text:
        return [args.text.strip()]

    if args.file:
        path = Path(args.file)
        if not path.exists():
            raise SystemExit(f"Файл {path} не найден")

        with path.open(encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
        if not texts:
            raise SystemExit("В указанном файле нет строк для анализа")
        return texts

    raise SystemExit("Передайте текст через --text или путь к файлу через --file")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Анализирует текст с использованием обученной baseline-модели."
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Одно сообщение для анализа.",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Файл с сообщениями (по одному в строке).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Порог риска для классификации (по умолчанию 0.7).",
    )
    args = parser.parse_args()

    texts = _load_texts(args)

    service = ConflictPredictionService(auto_train=True)
    print(f"[CLI] Модель готова, анализируем {len(texts)} строк...\n")

    for idx, text in enumerate(texts, start=1):
        result = service.predict_single(text, threshold=args.threshold)

        print(f"[{idx}] Сообщение:")
        print(text)
        print(
            f"  conflict_score = {result.conflict_score:.3f} "
            f"(risk: {result.risk_level}, threshold: {result.threshold})"
        )
        print("  probabilities:")
        for label, prob in result.labels.items():
            print(f"    - {label}: {prob:.3f}")
        print("-" * 40)


if __name__ == "__main__":
    main()
