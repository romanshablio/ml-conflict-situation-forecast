# ml-conflict-situation-forecast

Репозиторий для учебной практики и подготовки диплома по теме прогнозирования развития конфликтных ситуаций методами машинного и
нтеллекта: датасеты, эксперименты, прототипы моделей и вспомогательный код.

## Project layout

```
src/
├── data_ingestion.py     # CSV import and dataset merging helpers
├── preprocessing.py      # Cleaning and feature engineering for time series
└── models/
    └── training.py       # Chronological splits, training, and evaluation
```

Tests live in `tests/` and cover data ingestion, preprocessing, and modeling routines.

## Environment setup

1. Create a virtual environment (e.g., `python -m venv .venv` and `source .venv/bin/activate`).
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and adjust paths as needed.
4. Run the test suite with `pytest`.
