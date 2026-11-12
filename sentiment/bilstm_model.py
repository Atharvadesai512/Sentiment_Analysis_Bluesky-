from datasets import load_dataset
import pandas as pd

# 1. Download labelled sentiment dataset
dataset = load_dataset("cardiffnlp/tweet_eval", "sentiment")

# 2. Convert to DataFrames
train = dataset["train"].to_pandas()
validation = dataset["validation"].to_pandas()
test = dataset["test"].to_pandas()

# 3. Show dataset sizes
print("Training:", len(train))
print("Validation:", len(validation))
print("Testing:", len(test))

# 4. Show sentiment labels
print("\nLabels:")
print("0 = Negative")
print("1 = Neutral")
print("2 = Positive")

# 5. Show sample records
print("\nSample data:")
print(train.head())

# 6. Save datasets
train.to_csv("data/bilstm_train.csv", index=False)
validation.to_csv("data/bilstm_validation.csv", index=False)
test.to_csv("data/bilstm_test.csv", index=False)

print("\nDatasets saved successfully!")