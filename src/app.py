import os
import csv
import io
import json
from datetime import datetime
from functools import wraps

from flask import Flask, request, jsonify, send_from_directory, Response, session
import pandas as pd
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from .model_service import ConflictPredictionService
from .config import (
    REPORTS_DIR,
    BASE_DIR as PROJECT_BASE_DIR,
    CUSTOM_DATASET_PATH,
    BASELINE_MODEL_PATH,
    SECRET_KEY,
)
from .data_utils import TOXIC_LABELS, load_russian_toxic_data
from .visualization import save_label_distribution, save_roc_curves, save_text_report


app = Flask(__name__)
service = ConflictPredictionService() 
app.secret_key = SECRET_KEY

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = REPORTS_DIR / "history.json"
MODEL_META_FILE = REPORTS_DIR / "model_meta.json"


def _load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"[Service] Не удалось сохранить историю отчётов: {exc}")


def _save_report(report_data):
    report_id = report_data["id"]
    report_path = REPORTS_DIR / f"{report_id}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    history = _load_history()
    summary = {
        "id": report_id,
        "created_at": report_data["created_at"],
        "threshold": report_data["threshold"],
        "count": report_data["count"],
        "risk_distribution": report_data.get("risk_distribution", {}),
        "type": report_data.get("type", "batch"),
    }
    history.insert(0, summary)
    # ограничим историю, чтобы не раздувать файл
    history = history[:200]
    _save_history(history)

    return report_path


def _load_model_meta():
    if MODEL_META_FILE.exists():
        try:
            with open(MODEL_META_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_model_meta(meta: dict) -> None:
    try:
        with open(MODEL_META_FILE, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        print(f"[Service] Не удалось сохранить метаданные модели: {exc}")

CURRENT_MODEL_META = _load_model_meta() or {"id": "baseline", "artifacts": None, "metrics": None}


def _ensure_model_report() -> dict:
    """
    Если метаданные модели или артефакты отсутствуют, генерируем отчёт по текущей модели.
    Используем готовую загруженную модель и исходный датасет.
    """
    meta = _load_model_meta() or CURRENT_MODEL_META or {}
    meta.setdefault("id", "baseline")
    meta.setdefault("artifacts", {})
    meta.setdefault("metrics", {})

    artifacts = meta.get("artifacts") or {}
    # Если артефакты уже есть и файлы доступны — возвращаем
    if artifacts and all((REPORTS_DIR / fname).exists() for fname in artifacts.values() if fname):
        return meta

    if not service.is_ready():
        meta["error"] = "Модель не загружена."
        return meta

    try:
        X_train, X_test, y_train, y_test = load_russian_toxic_data()
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"Не удалось загрузить датасет: {exc}"
        return meta

    try:
        y_prob = service.model.predict_proba(X_test)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"Не удалось получить вероятности модели: {exc}"
        return meta

    label_dist_path = REPORTS_DIR / "model_label_dist.png"
    roc_path = REPORTS_DIR / "model_roc.png"
    text_report_path = REPORTS_DIR / "model_metrics.txt"

    try:
        save_label_distribution(pd.concat([y_train, y_test]), TOXIC_LABELS, label_dist_path)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"Ошибка сохранения распределения меток: {exc}"

    roc_auc = {}
    for i, lbl in enumerate(TOXIC_LABELS):
        try:
            if y_test.iloc[:, i].nunique() < 2:
                roc_auc[lbl] = None
                continue
            roc_auc[lbl] = float(roc_auc_score(y_test.iloc[:, i], y_prob[:, i]))
        except Exception:
            roc_auc[lbl] = None

    macro_auc = (
        float(np.mean([v for v in roc_auc.values() if v is not None]))
        if any(v is not None for v in roc_auc.values())
        else None
    )

    try:
        save_roc_curves(y_test.to_numpy(), y_prob, TOXIC_LABELS, roc_path)
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"Ошибка сохранения ROC-кривых: {exc}"

    try:
        save_text_report(
            {
                "n_samples": len(X_train) + len(X_test),
                "test_size": len(X_test),
                "roc_auc": roc_auc,
                "macro_auc": macro_auc,
                "notes": "Автоматически сгенерированный отчёт по текущей модели.",
            },
            text_report_path,
        )
    except Exception as exc:  # noqa: BLE001
        meta["error"] = f"Ошибка сохранения текстового отчёта: {exc}"

    new_meta = {
        "id": meta.get("id") or "baseline_autogen",
        "artifacts": {
            "label_distribution": label_dist_path.name if label_dist_path.exists() else None,
            "roc": roc_path.name if roc_path.exists() else None,
            "metrics": text_report_path.name if text_report_path.exists() else None,
        },
        "metrics": {"roc_auc": roc_auc, "macro_auc": macro_auc},
    }
    CURRENT_MODEL_META.update(new_meta)
    _save_model_meta(CURRENT_MODEL_META)
    return CURRENT_MODEL_META


def _build_pipeline():
    return Pipeline(
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
                        n_jobs=1,  # liblinear не использует >1, чтобы не ругался
                    )
                ),
            ),
        ]
    )


USERS = {
    "Admin": {"password": "pass_A2025", "role": "admin"},
    "Analyst": {"password": "pass_B2025", "role": "analyst"},
    "Chief": {"password": "pass_C2025", "role": "chief"},
}


def require_roles(*roles):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            user = session.get("user")
            role = session.get("role")
            if not user:
                return jsonify({"error": "Unauthorized"}), 401
            if roles and role not in roles:
                return jsonify({"error": "Forbidden"}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


@app.route("/")
def index():
    return send_from_directory(str(PROJECT_BASE_DIR), "index.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(force=True, silent=True) or {}
    login = data.get("login")
    password = data.get("password")
    user = USERS.get(login)
    if not user or user["password"] != password:
        return jsonify({"error": "Неверный логин или пароль"}), 401

    session["user"] = login
    session["role"] = user["role"]
    return jsonify({"user": login, "role": user["role"]})


@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"status": "ok"})


@app.route("/api/me", methods=["GET"])
def api_me():
    if "user" not in session:
        return jsonify({"authenticated": False}), 401
    return jsonify({"authenticated": True, "user": session["user"], "role": session.get("role")})


@app.route("/api/predict", methods=["POST"])
@require_roles("admin", "analyst")
def api_predict():
    """
    Ожидает JSON:
    {
      "text": "строка сообщения",
      "threshold": 0.7 (опционально)
    }
    """
    data = request.get_json(force=True, silent=True) or {}

    text = data.get("text", "")
    threshold = float(data.get("threshold", 0.7))

    if not text or not text.strip():
        return jsonify({"error": "Текст сообщения не должен быть пустым"}), 400

    try:
        # Больше НЕ передаём model_type, сервис всегда использует ML-модель
        result = service.predict_single_as_dict(text, threshold=threshold)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    score = result["conflict_score"]

    if score >= 0.8:
        risk_level = "high"
    elif score >= threshold:
        risk_level = "medium"
    else:
        risk_level = "low"

    response = {
        "text": result["text"],
        "model": "ml",  # фиксируем, что сейчас используется baseline ML
        "conflict_score": score,
        "risk_level": risk_level,
        "threshold": threshold,
        "labels": result["labels"],
    }
    return jsonify(response)


@app.route("/api/batch_predict", methods=["POST"])
@require_roles("admin", "analyst")
def api_batch_predict():
    """
    Ожидает JSON:
    {
      "texts": ["строка1", "строка2", ...],
      "threshold": 0.7 (опционально)
    }
    """
    data = request.get_json(force=True, silent=True) or {}

    texts = data.get("texts", [])
    threshold = float(data.get("threshold", 0.7))

    if not isinstance(texts, list) or not texts:
        return jsonify({"error": "Нужно передать непустой список 'texts'"}), 400

    try:
        # Без model_type, всегда используем baseline ML
        results = service.predict_batch(texts, threshold=threshold)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    enriched = []
    for r in results:
        score = r.conflict_score
        if score >= 0.8:
            risk_level = "high"
        elif score >= threshold:
            risk_level = "medium"
        else:
            risk_level = "low"

        enriched.append(
            {
                "text": r.text,
                "model": "ml",
                "conflict_score": score,
                "risk_level": risk_level,
                "labels": r.labels,
            }
        )

    return jsonify(
        {
            "threshold": threshold,
            "count": len(enriched),
            "results": enriched,
        }
    )


@app.route("/api/batch_predict_file", methods=["POST"])
@require_roles("admin", "analyst")
def api_batch_predict_file():
    """
    Принимает multipart/form-data с файлом.
    Поддерживаем:
    - CSV (ищем столбцы text/comment/comment_text/message/body, иначе берём первый)
    - TXT (по одному сообщению в строке)
    """
    if "file" not in request.files:
        return jsonify({"error": "Не передан файл 'file'"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Имя файла пустое"}), 400

    threshold = float(request.form.get("threshold", 0.7))
    filename = file.filename.lower()

    try:
        raw = file.read()
        if not raw:
            return jsonify({"error": "Файл пустой"}), 400
        try:
            text_data = raw.decode("utf-8")
        except UnicodeDecodeError:
            text_data = raw.decode("cp1251", errors="ignore")
    except Exception as e:  # noqa: BLE001
        return jsonify({"error": f"Не удалось прочитать файл: {e}"}), 400

    texts = []
    if filename.endswith(".csv"):
        buf = io.StringIO(text_data)
        reader = csv.DictReader(buf)
        if reader.fieldnames:
            preferred = ["text", "comment", "comment_text", "message", "body"]
            column = next((c for c in preferred if c in reader.fieldnames), reader.fieldnames[0])
            for row in reader:
                val = row.get(column, "")
                if val:
                    val = str(val).strip()
                    if val:
                        texts.append(val)
    else:
        # считаем, что это простой текстовый файл: одна строка — одно сообщение
        for line in text_data.splitlines():
            line = line.strip()
            if line:
                texts.append(line)

    if not texts:
        return jsonify({"error": "Не удалось найти текстовые сообщения в файле"}), 400

    # Ограничим размер, чтобы не перегружать демо-сервис
    max_items = 2000
    if len(texts) > max_items:
        texts = texts[:max_items]

    try:
        results = service.predict_batch(texts, threshold=threshold)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    enriched = []
    risk_counts = {"low": 0, "medium": 0, "high": 0}
    for r in results:
        risk_counts[r.risk_level] = risk_counts.get(r.risk_level, 0) + 1
        enriched.append(
            {
                "text": r.text,
                "model": "ml",
                "conflict_score": r.conflict_score,
                "risk_level": r.risk_level,
                "labels": r.labels,
            }
        )

    now = datetime.utcnow().isoformat() + "Z"
    report_id = now.replace(":", "").replace("-", "").replace(".", "")
    report_data = {
        "id": report_id,
        "created_at": now,
        "threshold": threshold,
        "count": len(enriched),
        "risk_distribution": risk_counts,
        "results": enriched,
        "type": "batch",
        "model_id": CURRENT_MODEL_META.get("id"),
        "artifacts": CURRENT_MODEL_META.get("artifacts"),
        "metrics": CURRENT_MODEL_META.get("metrics"),
    }
    _save_report(report_data)

    return jsonify(
        {
            "threshold": threshold,
            "count": len(enriched),
            "risk_distribution": risk_counts,
            "results": enriched,
            "report_id": report_id,
        }
    )


@app.route("/api/reports", methods=["GET"])
def api_reports():
    """
    Возвращает историю ранее сформированных отчётов.
    """
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    history = _load_history()
    return jsonify({"reports": history})


@app.route("/api/model_report", methods=["GET"])
def api_model_report():
    """
    Возвращает метаданные текущей модели и её артефакты (графики/отчёты).
    """
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    meta = _ensure_model_report()
    return jsonify({"model": meta})


@app.route("/report/<report_id>", methods=["GET"])
def get_report(report_id):
    """
    Возвращает HTML-страницу с подробным отчётом.
    """
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    report_path = REPORTS_DIR / f"{report_id}.json"
    if not report_path.exists():
        return jsonify({"error": "Отчёт не найден"}), 404

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    items_html = "".join(
        f"<li><strong>{idx+1}.</strong> {row['text']} — "
        f"риск: {row.get('risk_level','-')} (score: {row.get('conflict_score',0):.3f})</li>"
        for idx, row in enumerate(report.get("results", []))
    )

    artifacts = report.get("artifacts") or {}
    # Если графики не сохранены в самом отчёте, пробуем подтянуть их из отчёта обучения модели
    if (not artifacts) and report.get("model_id"):
        train_report = REPORTS_DIR / f"{report.get('model_id')}.json"
        if train_report.exists():
            try:
                with open(train_report, "r", encoding="utf-8") as tf:
                    tdata = json.load(tf)
                    artifacts = tdata.get("artifacts") or {}
                    # Заполним метрики, если их не было
                    if not report.get("metrics") and tdata.get("metrics"):
                        report["metrics"] = tdata["metrics"]
            except Exception:
                pass
    art_html = ""
    if artifacts:
        imgs = []
        links = []
        for name, fname in artifacts.items():
            if fname:
                links.append(f'<li>{name}: <a href="/report_file/{fname}" target="_blank">{fname}</a></li>')
                if fname.endswith((".png", ".jpg", ".jpeg")):
                    imgs.append(f'<div style="margin-top:10px;"><div class="muted">{name}</div><img src="/report_file/{fname}" alt="{name}" style="max-width:720px; width:100%; height:auto; border:1px solid #eee; border-radius:8px;"></div>')
        art_html = "<h3>Визуализации</h3>"
        if imgs:
            art_html += "".join(imgs)
        if links:
            art_html += "<ul>" + "".join(links) + "</ul>"

    html = f"""
    <html lang="ru">
      <head>
        <meta charset="UTF-8">
        <title>Отчёт {report_id}</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 16px; }}
          .pill {{ display: inline-block; padding: 4px 8px; border-radius: 12px; font-size: 12px; }}
          .pill-low {{ background:#e6f7ff; color:#096dd9; }}
          .pill-medium {{ background:#fff7e6; color:#d48806; }}
          .pill-high {{ background:#fff1f0; color:#cf1322; }}
          ul {{ padding-left: 18px; }}
        </style>
      </head>
      <body>
        <h2>Отчёт по пакетному анализу</h2>
        <div>Идентификатор: <code>{report_id}</code></div>
        <div>Создан: {report.get("created_at","")}</div>
        <div>Тип: {report.get("type","batch")}</div>
        <div>Модель: {report.get("model_id","n/a")}</div>
        <div>Объём: {report.get("count",0)}</div>
        <div>Порог риска: {report.get("threshold","—")}</div>
        {"<h3>Распределение по рискам</h3>" if report.get("type") in ("batch","quick") else ""}
        {"<div><span class='pill pill-low'>Низкий: "+str(report.get('risk_distribution',{}).get('low',0))+"</span> <span class='pill pill-medium'>Средний: "+str(report.get('risk_distribution',{}).get('medium',0))+"</span> <span class='pill pill-high'>Высокий: "+str(report.get('risk_distribution',{}).get('high',0))+"</span></div>" if report.get('type') in ('batch','quick') else ""}
        {"<h3>Детализация</h3><ul>"+items_html+"</ul>" if items_html else ""}
        {"<h3>Метрики</h3>" if report.get("metrics") else ""}
        {"<div>Macro AUC: "+str(report.get('metrics',{}).get('macro_auc'))+"</div>" if report.get("metrics") else ""}
        {art_html}
      </body>
    </html>
    """
    return Response(html, mimetype="text/html")


@app.route("/report_file/<path:fname>", methods=["GET"])
def get_report_file(fname):
    """
    Отдаёт сохранённые графики/отчёты из папки reports.
    """
    if "user" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    target = REPORTS_DIR / fname
    if not target.exists():
        return jsonify({"error": "Файл не найден"}), 404
    return send_from_directory(str(REPORTS_DIR), fname)


@app.route("/api/train_model", methods=["POST"])
@require_roles("admin")
def api_train_model():
    """
    Обучает ML-модель на загруженном датасете.
    Ожидается CSV-файл с колонкой текста (text/comment/comment_text/message/body)
    и бинарной колонкой токсичности (toxic/label).
    Сохраняет модель в models/baseline_model.pkl и обновляет сервис инференса.
    """
    if "file" not in request.files:
        return jsonify({"error": "Не передан файл 'file'"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Имя файла пустое"}), 400

    raw = file.read()
    if not raw:
        return jsonify({"error": "Файл пустой"}), 400
    try:
        text_data = raw.decode("utf-8")
    except UnicodeDecodeError:
        text_data = raw.decode("cp1251", errors="ignore")

    try:
        df = pd.read_csv(io.StringIO(text_data))
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": f"Не удалось прочитать CSV: {exc}"}), 400

    text_cols = [c for c in df.columns if c.lower() in {"text", "comment", "comment_text", "message", "body"}]
    label_cols = [c for c in df.columns if c.lower() in {"toxic", "label"}]
    if not text_cols or not label_cols:
        return jsonify({"error": "Нужны колонки с текстом (text/comment/...) и меткой токсичности (toxic/label)"}), 400

    text_col = text_cols[0]
    label_col = label_cols[0]

    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df = df[df[text_col] != ""]
    df[label_col] = df[label_col].astype(int)

    if df.empty:
        return jsonify({"error": "После очистки данных не осталось строк"}), 400

    # формируем мультилейбл-матрицу (если в датасете только toxic, остальное заполняем нулями)
    y = pd.DataFrame()
    y["toxic"] = df[label_col]
    for lbl in TOXIC_LABELS:
        if lbl not in y.columns:
            y[lbl] = y["toxic"] if lbl == "toxic" else 0
    y = y[TOXIC_LABELS]

    X = df[text_col]

    # Разделим данные для оценки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y["toxic"] if "toxic" in y else None
    )

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    # сохраняем датасет и модель
    CUSTOM_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CUSTOM_DATASET_PATH, index=False)
    joblib.dump(pipeline, BASELINE_MODEL_PATH)
    service.model = pipeline

    # Метрики и визуализация
    y_prob = pipeline.predict_proba(X_test)
    roc_auc = {}
    for i, lbl in enumerate(TOXIC_LABELS):
        try:
            roc_auc[lbl] = float(roc_auc_score(y_test.iloc[:, i], y_prob[:, i]))
        except ValueError:
            roc_auc[lbl] = None
    macro_auc = (
        float(np.mean([v for v in roc_auc.values() if v is not None]))
        if any(v is not None for v in roc_auc.values())
        else None
    )

    timestamp = datetime.utcnow().isoformat() + "Z"
    report_id = "training_" + timestamp.replace(":", "").replace("-", "").replace(".", "")

    label_dist_path = REPORTS_DIR / f"{report_id}_label_dist.png"
    roc_path = REPORTS_DIR / f"{report_id}_roc.png"
    text_report_path = REPORTS_DIR / f"{report_id}_metrics.txt"

    save_label_distribution(y, TOXIC_LABELS, label_dist_path)
    save_roc_curves(y_test.to_numpy(), y_prob, TOXIC_LABELS, roc_path)
    save_text_report(
        {
            "n_samples": len(df),
            "test_size": len(X_test),
            "roc_auc": roc_auc,
            "macro_auc": macro_auc,
            "notes": "Обучение базовой ML-модели на пользовательском датасете.",
        },
        text_report_path,
    )

    artifacts = {
        "label_distribution": label_dist_path.name,
        "roc": roc_path.name,
        "metrics": text_report_path.name,
    }
    report_data = {
        "id": report_id,
        "created_at": timestamp,
        "threshold": None,
        "count": len(df),
        "risk_distribution": {},
        "results": [],
        "type": "training",
        "artifacts": artifacts,
        "metrics": {"roc_auc": roc_auc, "macro_auc": macro_auc},
        "model_id": report_id,
    }
    _save_report(report_data)
    CURRENT_MODEL_META["id"] = report_id
    CURRENT_MODEL_META["artifacts"] = artifacts
    CURRENT_MODEL_META["metrics"] = {"roc_auc": roc_auc, "macro_auc": macro_auc}
    _save_model_meta(CURRENT_MODEL_META)

    return jsonify(
        {
            "status": "ok",
            "trained_on": len(df),
            "text_column": text_col,
            "label_column": label_col,
            "model_path": str(BASELINE_MODEL_PATH),
            "report_id": report_id,
            "artifacts": {
                "label_distribution": label_dist_path.name,
                "roc": roc_path.name,
                "metrics": text_report_path.name,
            },
            "model_id": report_id,
        }
    )


@app.route("/api/save_quick", methods=["POST"])
@require_roles("admin", "analyst")
def api_save_quick():
    """
    Сохраняет результат быстрого анализа как отдельный отчёт (1 сообщение).
    Ожидает JSON: { "text": "...", "threshold": 0.7 }
    """
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    threshold = float(data.get("threshold", 0.7))

    if not text or not text.strip():
        return jsonify({"error": "Текст сообщения не должен быть пустым"}), 400

    try:
        result = service.predict_single(text, threshold=threshold)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    risk_counts = {"low": 0, "medium": 0, "high": 0}
    risk_counts[result.risk_level] = 1

    now = datetime.utcnow().isoformat() + "Z"
    report_id = now.replace(":", "").replace("-", "").replace(".", "")
    report_data = {
        "id": report_id,
        "created_at": now,
        "threshold": threshold,
        "count": 1,
        "risk_distribution": risk_counts,
        "results": [
            {
                "text": result.text,
                "model": "ml",
                "conflict_score": result.conflict_score,
                "risk_level": result.risk_level,
                "labels": result.labels,
            }
        ],
        "type": "quick",
        "model_id": CURRENT_MODEL_META.get("id"),
        "artifacts": CURRENT_MODEL_META.get("artifacts"),
        "metrics": CURRENT_MODEL_META.get("metrics"),
    }
    _save_report(report_data)

    return jsonify(
        {
            "status": "saved",
            "report_id": report_id,
            "risk_level": result.risk_level,
            "conflict_score": result.conflict_score,
        }
    )


if __name__ == "__main__":
    # локальный запуск backend для теста
    app.run(host="0.0.0.0", port=5500, debug=True)
