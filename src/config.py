import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "train.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pkl")
DL_MODEL_PATH = os.path.join(MODELS_DIR, "dl_model.h5")
BASELINE_MODEL_PATH = os.path.join(MODELS_DIR, "baseline_model.pkl")

# Настройки DL-модели
MAX_NUM_WORDS = 30000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 128
BATCH_SIZE = 64
EPOCHS = 5  # потом можно увеличить
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42