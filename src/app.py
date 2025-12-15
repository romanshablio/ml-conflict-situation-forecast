import os
import csv
import io
import json
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, Response

from .model_service import ConflictPredictionService
from .config import REPORTS_DIR, BASE_DIR as PROJECT_BASE_DIR


app = Flask(__name__)
service = ConflictPredictionService() 

REPORTS_DIR.mkdir(parents=True, exist_ok=True)

HISTORY_FILE = REPORTS_DIR / "history.json"


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
    }
    history.insert(0, summary)
    # ограничим историю, чтобы не раздувать файл
    history = history[:200]
    _save_history(history)

    return report_path

@app.route("/")
def index():
    return send_from_directory(str(PROJECT_BASE_DIR), "index.html")


@app.route("/api/predict", methods=["POST"])
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
    history = _load_history()
    return jsonify({"reports": history})


@app.route("/report/<report_id>", methods=["GET"])
def get_report(report_id):
    """
    Возвращает HTML-страницу с подробным отчётом.
    """
    report_path = REPORTS_DIR / f"{report_id}.json"
    if not report_path.exists():
        return jsonify({"error": "Отчёт не найден"}), 404

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    items_html = "".join(
        f"<li><strong>{idx+1}.</strong> {row['text']} — "
        f"риск: {row['risk_level']} (score: {row['conflict_score']:.3f})</li>"
        for idx, row in enumerate(report.get("results", []))
    )

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
        <div>Сообщений: {report.get("count",0)}</div>
        <div>Порог риска: {report.get("threshold",0.7)}</div>
        <h3>Распределение по рискам</h3>
        <div>
          <span class="pill pill-low">Низкий: {report.get("risk_distribution",{}).get("low",0)}</span>
          <span class="pill pill-medium">Средний: {report.get("risk_distribution",{}).get("medium",0)}</span>
          <span class="pill pill-high">Высокий: {report.get("risk_distribution",{}).get("high",0)}</span>
        </div>
        <h3>Детализация</h3>
        <ul>{items_html}</ul>
      </body>
    </html>
    """
    return Response(html, mimetype="text/html")


if __name__ == "__main__":
    # локальный запуск backend для теста
    app.run(host="0.0.0.0", port=5500, debug=True)
