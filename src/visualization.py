from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional, Dict, Any

import matplotlib

# Используем нерисующий бэкенд, чтобы не требовался GUI (важно для серверов/macOS без main thread).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


def save_label_distribution(y: pd.DataFrame, labels: Sequence[str], path: Path) -> Path:
    """
    Строит бар-чарт распределения меток токсичности.
    """
    counts = {label: int(y[label].sum()) for label in labels if label in y.columns}
    if not counts:
        return path

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(counts.keys(), counts.values(), color="#3f73ff", alpha=0.9)
    ax.set_ylabel("Кол-во примеров")
    ax.set_title("Распределение меток в датасете")
    for idx, (label, val) in enumerate(counts.items()):
        ax.text(idx, val, str(val), ha="center", va="bottom", fontsize=9)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    labels: Sequence[str],
    path: Path,
) -> Path:
    """
    Строит ROC-кривые для каждого класса.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    for i, label in enumerate(labels):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")
        except ValueError:
            # Если класс отсутствует, пропускаем
            continue

    ax.plot([0, 1], [0, 1], "k--", label="No skill")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC-кривые")
    ax.legend(loc="lower right", fontsize=8)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_text_report(
    report: Dict[str, Any],
    path: Path,
) -> Path:
    """
    Сохраняет текстовый отчёт с метриками.
    """
    lines = []
    lines.append("Отчёт об обучении модели")
    lines.append(f"Всего примеров: {report.get('n_samples')}")
    lines.append(f"Размер теста: {report.get('test_size')}")
    lines.append("")
    lines.append("ROC-AUC по меткам:")
    for lbl, val in report.get("roc_auc", {}).items():
        lines.append(f"- {lbl}: {val:.3f}")
    if report.get("macro_auc") is not None:
        lines.append(f"Macro AUC: {report['macro_auc']:.3f}")
    lines.append("")
    if "notes" in report:
        lines.append("Комментарий:")
        lines.append(report["notes"])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
