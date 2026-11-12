
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load saved model and test dataset
model = tf.keras.models.load_model("models/bilstm_model.keras")
test = pd.read_csv("data/bilstm_test.csv").dropna()

x_test = test["text"].astype(str).to_numpy()
y_test = test["label"].astype(int).to_numpy()

# Predict sentiment
probabilities = model.predict(x_test, batch_size=64)
predictions = np.argmax(probabilities, axis=1)

# Evaluate
print("Test Accuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1, 2],
    target_names=["Negative", "Neutral", "Positive"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

# Save predictions
test["bilstm_sentiment"] = [
    ["negative", "neutral", "positive"][i]
    for i in predictions
]
test.to_csv("data/bilstm_test_results.csv", index=False)

print("\nResults saved successfully!")
