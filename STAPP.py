from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

ARTIFACT = "knn_pass_fail_pipeline.pkl"
DEFAULT_THRESHOLD = 0.50

# EXACT training feature names (model expects these)
FEATURES = [
    "Attendance (%)",
    "Quizzes_Avg",
    "Assignments_Avg",
    "Midterm_Score",
    "Participation_Score",
    "Projects_Score",
    "Study_Hours_per_Week",
    "Sleep_Hours_per_Night",
    "Stress_Level (1-10)",
    "Extracurricular_Activities"
]

# Load pre-trained model
if not os.path.exists(ARTIFACT):
    raise SystemExit("Missing 'knn_pass_fail_pipeline.pkl'. Train first.")
pipe = joblib.load(ARTIFACT)

def final_estimator(pipeline):
    if hasattr(pipeline, "named_steps") and pipeline.named_steps:
        return list(pipeline.named_steps.values())[-1]
    return pipeline

def detect_indices_for_pass_fail(pipeline):
    est = final_estimator(pipeline)
    if not hasattr(est, "classes_"):
        raise RuntimeError("Final estimator has no classes_.")
    classes = np.array(est.classes_)
    up = np.array([str(c).upper() for c in classes])
    if "PASS" in up and "FAIL" in up:
        p_idx = int(np.where(up == "PASS")[0][0])
        f_idx = int(np.where(up == "FAIL")[0][0])
        return classes, p_idx, f_idx
    if set(classes) == {0, 1} or set(classes) == {0., 1.}:
        return classes, 0, 1  # assume 0=PASS, 1=FAIL
    raise RuntimeError(f"Cannot infer PASS/FAIL mapping from classes: {classes}")

CLASSES, PASS_IDX, FAIL_IDX = detect_indices_for_pass_fail(pipe)

@app.route('/')
def index():
    return render_template('index.html', threshold=DEFAULT_THRESHOLD)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Build row matching the exact feature order
        row = {
            "Attendance (%)": float(data.get("Attendance (%)", 85)),
            "Quizzes_Avg": float(data.get("Quizzes_Avg", 75)),
            "Assignments_Avg": float(data.get("Assignments_Avg", 78)),
            "Midterm_Score": float(data.get("Midterm_Score", 70)),
            "Participation_Score": float(data.get("Participation_Score", 80)),
            "Projects_Score": float(data.get("Projects_Score", 82)),
            "Study_Hours_per_Week": float(data.get("Study_Hours_per_Week", 12)),
            "Sleep_Hours_per_Night": float(data.get("Sleep_Hours_per_Night", 7.0)),
            "Stress_Level (1-10)": float(data.get("Stress_Level (1-10)", 5)),
            "Extracurricular_Activities": str(data.get("Extracurricular_Activities", "Yes"))
        }
        
        # Create DataFrame with exact column order
        X = pd.DataFrame([row], columns=FEATURES)
        
        # Predict probabilities
        proba = pipe.predict_proba(X)[0]
        p_pass = float(proba[PASS_IDX])
        p_fail = float(proba[FAIL_IDX])
        
        # Get threshold
        threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
        
        # Calculate prediction
        prediction = "PASS" if p_pass >= threshold else "FAIL"
        expected_grades = "A, B, or C" if prediction == "PASS" else "D or F"
        
        result = {
            "prediction": prediction,
            "pass_prob": round(p_pass * 100, 1),
            "fail_prob": round(p_fail * 100, 1),
            "expected_grades": expected_grades
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)