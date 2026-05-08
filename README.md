# ml-conflict-situation-forecast

A machine learning prototype for forecasting conflict situations from text messages.

The project contains an intelligent text analysis prototype, a baseline machine learning model for toxicity and conflict estimation, a Flask inference API, and a simple HTML/JavaScript demo interface. Some parts, including a deep learning model and extended analytics, are planned as part of the graduation thesis work.

## Project Contents

- Intelligent text message analysis prototype.
- Baseline machine learning model for toxicity and conflict detection.
- Flask API for inference.
- Web interface for demonstration.
- Future placeholders for a deep learning model and extended analytics.

## Project Structure

```text
ml-conflict-situation-forecast/
|-- data/
|   |-- ru_toxic_2ch_pikabu.csv
|   +-- ru_toxic_ok.txt
|-- models/
|   |-- baseline_model.pkl
|   |-- tokenizer.pkl
|   +-- dl_model.h5
|-- src/
|   |-- config.py
|   |-- data_utils.py
|   |-- baseline_model.py
|   |-- dl_model.py
|   |-- model_service.py
|   +-- app.py
|-- index.html
|-- requirements.txt
+-- README.md
```

The current version uses a baseline ML model. Deep learning related files are placeholders for future development.

## Requirements

- Python 3.9 or newer.
- Virtual environment such as `venv`.
- External Russian-language toxicity datasets from Kaggle or Google Drive.
- macOS, Linux, or Windows.

## Quick Start

1. Clone the repository.

   ```bash
   git clone https://github.com/romanshablio/ml-conflict-situation-forecast.git
   cd ml-conflict-situation-forecast
   ```

2. Create and activate a virtual environment.

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   # Windows PowerShell:
   # venv\Scripts\activate
   ```

3. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

4. Prepare external files.

   The project uses two open Russian-language toxic comment datasets:

   - 2ch.hk + Pikabu.ru dataset in CSV format.
   - ok.ru dataset in fastText TXT format.

   Because of licensing and file size, the raw datasets and trained models are not stored directly in this GitHub repository.

   For review, the required files are available in a shared Google Drive folder:

   ```text
   https://drive.google.com/drive/folders/1J-vftGANncLTPWcX7b7MGhfnJwSfVZnd?usp=sharing
   ```

   Place the downloaded files into:

   ```text
   data/ru_toxic_2ch_pikabu.csv
   data/ru_toxic_ok.txt
   models/baseline_model.pkl
   ```

5. Train the baseline model if needed.

   ```bash
   python -m src.baseline_model
   ```

   This creates:

   ```text
   models/baseline_model.pkl
   ```

6. Start the web application.

   ```bash
   python main.py
   ```

   Or:

   ```bash
   python -m src.app
   ```

   The application starts on port `5500` by default:

   ```text
   http://127.0.0.1:5500/
   ```

## Main Components

### `baseline_model.py`

- Loads and combines Russian-language toxic comment datasets.
- Builds a TF-IDF plus `OneVsRestClassifier(LogisticRegression)` pipeline.
- Trains a binary toxic/non-toxic model.
- Saves the model to `models/baseline_model.pkl`.
- Prints basic quality metrics.

### `model_service.py`

- Loads the trained baseline model.
- Provides prediction methods:
  - `predict_single(text: str) -> dict`
  - `predict_batch(texts: List[str]) -> List[dict]`
- Returns toxicity probability, conflict score, and risk level.

### `app.py`

- Starts the Flask application.
- Serves `index.html`.
- Provides the `POST /api/predict` endpoint.

Example response:

```json
{
  "text": "...",
  "model": "ml_ru",
  "conflict_score": 0.42,
  "risk_level": "medium",
  "threshold": 0.7,
  "labels": {
    "toxic": 0.42
  }
}
```

## Current Limitations and Plans

- The baseline model is trained on noisy Russian-language toxicity datasets from 2ch, Pikabu, and ok.ru.
- False positives and classification errors are possible.
- The prototype demonstrates architecture and working mechanics rather than production accuracy.
- Planned work includes better model quality, deep learning integration, batch analysis, run history, and richer reports.
