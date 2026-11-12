
import os
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense, Dropout
)
from tensorflow.keras.layers import TextVectorization

# 1. Load labelled data
train = pd.read_csv("data/bilstm_train.csv").dropna()
val = pd.read_csv("data/bilstm_validation.csv").dropna()

x_train = train["text"].astype(str).to_numpy()
y_train = train["label"].astype(int).to_numpy()

x_val = val["text"].astype(str).to_numpy()
y_val = val["label"].astype(int).to_numpy()

# 2. Convert text into numbers
vectorizer = TextVectorization(
    max_tokens=20000,
    output_mode="int",
    output_sequence_length=100
)
vectorizer.adapt(x_train)

# 3. Build BiLSTM model
model = Sequential([
    Input(shape=(), dtype=tf.string),
    vectorizer,
    Embedding(input_dim=20000, output_dim=64, mask_zero=True),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dense(3, activation="softmax")
])

# 4. Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    x_train,
    y_train,
    validation_data=(x_val, y_val),
    epochs=10,
    batch_size=64,
    callbacks=[early_stop]
)

# 6. Save model
os.makedirs("models", exist_ok=True)
model.save("models/bilstm_model.keras")

print("BiLSTM training completed!")
print("Model saved successfully!")
