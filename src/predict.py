import os
import pandas as pd
import joblib

print("🚀 Starting prediction...")

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

all_data = []

for file in os.listdir("data/test"):
    if file.endswith(".csv"):
        print(f"Reading {file}")
        df = pd.read_csv(f"data/test/{file}")
        all_data.append(df)

data = pd.concat(all_data, ignore_index=True)

X = data["text"]
X_vec = vectorizer.transform(X)

predictions = model.predict(X_vec)

output = pd.DataFrame({
    "id": data["id"],
    "polarization": predictions
})

output.to_csv("outputs/submission.csv", index=False)

print("✅ Predictions saved!")