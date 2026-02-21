from flask import Flask, render_template, request
import pickle
import numpy as np
import re
from scipy.sparse import hstack

# ----------------------------
# CREATE FLASK APP FIRST
# ----------------------------
app = Flask(__name__)

# ----------------------------
# LOAD MODEL
# ----------------------------
model = pickle.load(open("phishing_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ----------------------------
# FEATURE FUNCTION
# ----------------------------
def extract_features(url):
    return [
        len(url),
        sum(c.isdigit() for c in url),
        url.count('.'),
        url.count('-'),
        int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', url)))
    ]

# ----------------------------
# HOME ROUTE
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ----------------------------
# PREDICT ROUTE
# ----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    url = request.form["url"]

    url_vector = vectorizer.transform([url])
    extra = np.array([extract_features(url)])

    final_input = hstack([url_vector, extra])

    prediction = model.predict(final_input)[0]
    probability = model.predict_proba(final_input)[0]
    confidence = round(np.max(probability) * 100, 2)

    if prediction == 0:
        result = "⚠️ Phishing Website"
    else:
        result = "✅ Safe Website"

    return render_template(
        "index.html",
        prediction_text=result,
        confidence_text=f"Confidence: {confidence}%"
    )

# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)