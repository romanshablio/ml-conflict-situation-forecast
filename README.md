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

## Development workflow

Common tasks are available through the `Makefile`:

* `make setup` – install dependencies from `requirements.txt`.
* `make lint` – run Ruff, Flake8, and a Black formatting check.
* `make test` – execute the pytest suite.
* `make run` – launch the demo CLI (`python -m src.cli`).

Pre-commit hooks are configured to enforce formatting and linting. Install them with
`pre-commit install` after setting up the virtual environment.
