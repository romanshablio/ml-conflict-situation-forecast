import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

from .config import (
    MODELS_DIR,
    DL_MODEL_PATH,
    MAX_NUM_WORDS,
    MAX_SEQUENCE_LENGTH,
    EMBEDDING_DIM,
    BATCH_SIZE,
    EPOCHS,
)
from .data_utils import (
    load_raw_data,
    train_tokenizer,
    texts_to_padded_sequences,
    train_val_test_split,
    TOXIC_LABELS,
)


def build_dl_model(num_labels: int) -> Sequential:
    model = Sequential()
    model.add(
        Embedding(
            input_dim=MAX_NUM_WORDS,
            output_dim=EMBEDDING_DIM,
            input_length=MAX_SEQUENCE_LENGTH,
        )
    )
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_labels, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    return model


def train_dl_model():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Загружаем данные...")
    df = load_raw_data()
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(df)

    print("Обучаем токенайзер...")
    tokenizer = train_tokenizer(X_train)

    print("Преобразуем тексты в последовательности...")
    X_train_seq = texts_to_padded_sequences(X_train, tokenizer)
    X_val_seq = texts_to_padded_sequences(X_val, tokenizer)
    X_test_seq = texts_to_padded_sequences(X_test, tokenizer)

    num_labels = len(TOXIC_LABELS)
    model = build_dl_model(num_labels)

    callbacks = [
      EarlyStopping(
          monitor="val_loss",
          patience=2,
          restore_best_weights=True,
      ),
      ModelCheckpoint(
          filepath=DL_MODEL_PATH,
          monitor="val_loss",
          save_best_only=True,
      ),
    ]

    print("Начинаем обучение модели...")
    history = model.fit(
        X_train_seq,
        y_train,
        validation_data=(X_val_seq, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print("Оцениваем на тестовой выборке...")
    test_loss, test_acc = model.evaluate(X_test_seq, y_test, verbose=0)
    print(f"Тестовая loss: {test_loss:.4f}, accuracy: {test_acc:.4f}")

    # Сохраняем финальную модель (лучший чекпоинт уже сохранён)
    model.save(DL_MODEL_PATH)
    print(f"Модель сохранена в {DL_MODEL_PATH}")

    return history, (test_loss, test_acc)


if __name__ == "__main__":
    train_dl_model()