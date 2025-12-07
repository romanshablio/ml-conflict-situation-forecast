.PHONY: setup lint test run

setup:
python -m pip install -r requirements.txt

lint:
ruff check .
flake8 src tests
black --check .

test:
pytest

run:
python -m src.cli
