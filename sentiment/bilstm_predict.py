
import pandas as pd
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("models/bilstm_model.keras")

# Load Bluesky posts
df = pd.read_csv("data/cleaned_posts.csv")

# Predict in batches
texts = df["text"].fillna("").astype(str).to_numpy()
probs = model.predict(texts, batch_size=64, verbose=1)

# Convert predictions into sentiment labels
labels = np.array(["negative", "neutral", "positive"])
df["bilstm_sentiment"] = labels[np.argmax(probs, axis=1)]

# Save results
df.to_csv("data/bilstm_results.csv", index=False)

print("Total classified:", len(df))
print(df["bilstm_sentiment"].value_counts())
print("Saved: data/bilstm_results.csv")
