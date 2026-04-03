from flask import Flask, request, jsonify, render_template_string
import joblib

app = Flask(__name__)

model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Simple HTML UI
html = """
<!DOCTYPE html>
<html>
<head>
    <title>Polarization Detector</title>
</head>
<body>
    <h2>Enter Text:</h2>
    <form method="post" action="/predict_ui">
        <input type="text" name="text" style="width:300px;">
        <button type="submit">Predict</button>
    </form>
    <h3>{{ result }}</h3>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(html)

@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    text = request.form["text"]
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]

    result = "Non-Polarized" if pred == 1 else "Polarized"
    return render_template_string(html, result=result)

if __name__ == "__main__":
    app.run(debug=True)