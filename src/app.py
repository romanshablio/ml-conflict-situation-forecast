import os
from flask import Flask, request, jsonify, send_from_directory
from flask import render_template_string

from .model_service import ConflictPredictionService


app = Flask(__name__)
service = ConflictPredictionService() 

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


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
        result = service.predict_single(text)
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
        results = service.predict_batch(texts)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    enriched = []
    for r in results:
        score = r["conflict_score"]
        if score >= 0.8:
            risk_level = "high"
        elif score >= threshold:
            risk_level = "medium"
        else:
            risk_level = "low"

        enriched.append(
            {
                "text": r["text"],
                "model": "ml",
                "conflict_score": score,
                "risk_level": risk_level,
                "labels": r["labels"],
            }
        )

    return jsonify(
        {
            "threshold": threshold,
            "count": len(enriched),
            "results": enriched,
        }
    )


if __name__ == "__main__":
    # локальный запуск backend для теста
    app.run(host="0.0.0.0", port=5500, debug=True)