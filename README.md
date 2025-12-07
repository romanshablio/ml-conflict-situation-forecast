# ml-conflict-situation-forecast

Репозиторий для учебной практики и подготовки диплома по теме прогнозирования развития конфликтных ситуаций методами машинного и
нтеллекта: датасеты, эксперименты, прототипы моделей и вспомогательный код.

## Project layout

```
conflict_prediction/
├─ data/
│  └─ train.csv                 # датасет Jigsaw (скачиваешь с Kaggle и кладёшь сюда)
├─ models/
│  ├─ tokenizer.pkl             # сохранённый токенайзер (после обучения)
│  ├─ dl_model.h5               # обученная DL-модель (Keras)
│  └─ baseline_model.pkl        # ML-модель
├─ notebooks/
│  └─ experiments.ipynb         # эксперименты (по желанию)
├─ src/
│  ├─ __init__.py
│  ├─ config.py                 # общие настройки
│  ├─ data_utils.py             # загрузка/подготовка данных
│  ├─ dl_model.py               # архитектура и обучение DL-модели
│  ├─ baseline_model.py         # baseline ML-модель
│  ├─ model_service.py          # обёртка инференса (загрузка моделей, предсказание)
│  └─ app.py                    # Flask-приложение (API для интерфейса)
├─ requirements.txt
└─ README.md
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

# ml-conflict-situation-forecast

Репозиторий для учебной преддипломной практики и подготовки ВКР по теме  
«Прогнозирование развития конфликтных ситуаций методами машинного интеллекта».

Проект содержит:

- прототип интеллектуальной системы анализа текстовых сообщений;
- baseline‑модель машинного обучения для оценки токсичности/конфликтности;
- Flask‑API для инференса;
- веб‑интерфейс (HTML/JS) для демонстрации работы системы.

Часть функциональности (DL‑модель, полноценное обучение и расширенная аналитика) запланирована к реализации в рамках ВКР.

---

## Структура проекта

```
ml-conflict-situation-forecast/
├─ data/
│  └─ train.csv                 # датасет Jigsaw (скачивается отдельно и кладётся сюда)
├─ models/
│  ├─ baseline_model.pkl        # обученная baseline ML‑модель (создаётся после обучения)
│  ├─ tokenizer.pkl             # задел под DL‑модель (будет использоваться в ВКР)
│  └─ dl_model.h5               # задел под DL‑модель (будет использоваться в ВКР)
├─ src/
│  ├─ __init__.py
│  ├─ config.py                 # общие настройки путей и параметров
│  ├─ data_utils.py             # загрузка и разбиение данных для моделей ML
│  ├─ baseline_model.py         # обучение baseline‑модели (TF‑IDF + LogisticRegression)
│  ├─ dl_model.py               # архитектура и обучение DL‑модели (проект, на будущее)
│  ├─ model_service.py          # сервис инференса поверх baseline‑модели
│  └─ app.py                    # Flask‑приложение, REST‑API и отдача index.html
├─ index.html                   # прототип веб‑интерфейса (быстрый анализ, пакетный анализ и пр.)
├─ requirements.txt             # зависимости Python
└─ README.md
```

> Примечание: в текущей версии используется **baseline‑модель ML**; файлы, связанные с DL‑моделью, являются заделом для дальнейшей доработки в рамках ВКР.

---

## Требования

- Python 3.9+  
- виртуальное окружение (`venv` или аналог);  
- доступ к интернету для первоначального скачивания датасета Jigsaw (через браузер);  
- ОС: macOS / Linux / Windows.

---

## Быстрый старт (для проверки преподавателем)

1. **Клонируйте репозиторий**

   ```bash
   git clone https://github.com/&lt;user&gt;/ml-conflict-situation-forecast.git
   cd ml-conflict-situation-forecast
   ```

2. **Создайте и активируйте виртуальное окружение**

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # для macOS / Linux
   # venv\Scripts\activate         # для Windows PowerShell
   ```

3. **Установите зависимости**

   ```bash
   pip install -r requirements.txt
   ```

4. **Скачайте датасет Jigsaw и поместите его в `data/`**

   Используется открытый датасет *Jigsaw Toxic Comment Classification Challenge*  
   (Jigsaw / Kaggle / HuggingFace). Необходимо скачать файл `train.csv` и сохранить его по пути:

   ```text
   ml-conflict-situation-forecast/data/train.csv
   ```

5. **Обучите baseline‑модель**

   ```bash
   (venv) python -m src.baseline_model
   ```

   После выполнения в каталоге `models/` появится файл:

   ```text
   models/baseline_model.pkl
   ```

6. **Запустите веб‑приложение**

   ```bash
   (venv) python -m src.app
   ```

   По умолчанию приложение стартует на порту **5500**.  
   Откройте в браузере:

   ```
   http://127.0.0.1:5500/
   ```

   На экране «Быстрый анализ» можно ввести текст сообщения и получить оценку вероятности
   конфликтной ситуации. Запросы отправляются на endpoint:

   - `POST /api/predict` — быстрый анализ одного сообщения.

---

## Основные компоненты

### `baseline_model.py`

- загружает данные из `data/train.csv` через `data_utils.py`;
- формирует ML‑pipeline (TF‑IDF + `OneVsRestClassifier(LogisticRegression)`);
- обучает модель и сохраняет её в `models/baseline_model.pkl`;
- выводит базовые метрики качества (classification report) в консоль.

### `model_service.py`

- загружает обученную baseline‑модель;
- предоставляет методы:
  - `predict_single(text: str) -> dict`
  - `predict_batch(texts: List[str]) -> List[dict]`
- возвращает:
  - вероятности по классам токсичности (toxic, insult, threat, obscene, severe_toxic, identity_hate);
  - агрегированный `conflict_score` и уровень риска.

### `app.py`

- инициализирует Flask‑приложение;
- отдаёт `index.html` по корневому маршруту `/`;
- реализует REST‑endpoint:

  - `POST /api/predict` — анализ одного сообщения (используется веб‑интерфейсом).

Ответ имеет вид:

```json
{
  "text": "...",
  "model": "ml",
  "conflict_score": 0.42,
  "risk_level": "medium",
  "threshold": 0.7,
  "labels": {
    "toxic": 0.42,
    "insult": 0.21,
    "threat": 0.03,
    "obscene": 0.15,
    "severe_toxic": 0.05,
    "identity_hate": 0.01
  }
}
```

### `index.html`

- прототип интерфейса системы:
  - «Быстрый анализ сообщения» — **подключён к backend** и использует `/api/predict`;
  - «Пакетный анализ», «История», «Обучение модели» — демонстрационные экраны
    с заглушками, которые планируется доработать в ВКР.
- вся верстка и тексты интерфейса — на русском языке, в соответствии с требованиями к отчёту.

---

## Ограничения и планы развития

- baseline‑модель обучена на англоязычном датасете Jigsaw, поэтому качество анализа
  русскоязычных сообщений ограничено; прототип используется для демонстрации
  архитектуры и рабочих механизмов системы.
- в рамках ВКР планируется:
  - адаптация модели под русскоязычные данные;
  - обучение и интеграция DL‑модели (LSTM/GRU или трансформер);
  - реализация реального пакетного анализа и истории запусков;
  - расширение отчётности и метрик.

Этот README отражает текущее состояние проекта и может использоваться преподавателем
для развертывания и проверки работоспособности прототипа.