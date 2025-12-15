from pathlib import Path

# Корневой каталог проекта
BASE_DIR = Path(__file__).resolve().parent.parent

# Папка с данными и моделями
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
CUSTOM_DATASET_PATH = DATA_DIR / "custom_dataset.csv"

# Пути к русскоязычным датасетам
# Переименуй свои файлы в соответствии с этими именами
RU_2CH_PIKABU_PATH = DATA_DIR / "ru_toxic_2ch_pikabu.csv"
RU_OK_DATA_PATH = DATA_DIR / "ru_toxic_ok.txt"

# Путь для сохранения baseline-модели
BASELINE_MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
