import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("🚀 Starting training...")

all_data = []

for file in os.listdir("data/train"):
    if file.endswith(".csv"):
        print(f"Reading {file}")
        df = pd.read_csv(f"data/train/{file}")
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

X = data["text"]
y = data["polarization"]

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=200)
model.fit(X_vec, y)

joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("✅ Training complete!")