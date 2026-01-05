from flask import Flask, render_template, request, jsonify
import joblib
from model.preprocess import preprocess_text

app = Flask(__name__, static_url_path="/static", template_folder="templates")

# Load artifacts
tfidf = joblib.load("artifacts/tfidf.pkl")
svd = joblib.load("artifacts/svd.pkl")
le = joblib.load("artifacts/label_encoder.pkl")
clf = joblib.load("artifacts/model.pkl")

# Session aggregation
AGG = {"valid": 0, "invalid": 0, "total": 0}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    summary = request.form.get("Summary", "")
    clean = preprocess_text(summary)
    X_vec = tfidf.transform([clean])
    X = svd.transform(X_vec)

    # Predict numeric label, then decode to string
    pred_num = clf.predict(X)[0]                     # numeric (0 or 1)
    pred = le.inverse_transform([pred_num])[0]       # decode to "Valid"/"Invalid"

    probs = clf.predict_proba(X)[0]
    pred_index = list(le.classes_).index(pred)       # find index of predicted class
    confidence = float(probs[pred_index])

    # Update session stats
    AGG["total"] += 1
    if pred == "Valid":
        AGG["valid"] += 1
    else:
        AGG["invalid"] += 1

    valid_pct = (AGG["valid"] / AGG["total"]) * 100 if AGG["total"] else 0.0
    invalid_pct = (AGG["invalid"] / AGG["total"]) * 100 if AGG["total"] else 0.0

    return jsonify({
        "Status": pred,
        "Confidence": f"{confidence*100:.2f}%",
        "Valid_Percentage": f"{valid_pct:.2f}%",
        "Invalid_Percentage": f"{invalid_pct:.2f}%"
    })

if __name__ == "__main__":
    app.run(debug=True, port=8000)  # runs on http://127.0.0.1:8000