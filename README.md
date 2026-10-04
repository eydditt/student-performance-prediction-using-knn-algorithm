# Student Performance Prediction System (KNN)

An end-to-end machine learning web application built to predict university student academic outcomes (PASS/FAIL) and facilitate early academic intervention. The system integrates traditional academic metrics with behavioral data (stress levels, sleep patterns) to identify at-risk students before final examinations.

---

## 🛠 Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, Pandas, NumPy, Imbalanced-learn (SMOTE)
* **Web Framework:** Flask
* **Frontend:** HTML, CSS

---

## 📊 Model Architecture & Optimization
The prediction engine utilizes the **K-Nearest Neighbors (KNN)** algorithm, optimized through extensive hyperparameter tuning (`GridSearchCV`) to prioritize the detection of failing students.
* **Optimal K-Value:** 15
* **Distance Metric:** Manhattan Distance
* **Class Balancing:** Synthetic Minority Over-sampling Technique (SMOTE) applied to correct the 60:40 PASS/FAIL imbalance.
* **Feature Scaling:** Min-Max Scaler

---

## 🚀 Key Results
The model was evaluated with a strict focus on **Recall** (Sensitivity) for the 'FAIL' class to ensure at-risk students are not overlooked.
* **Overall Accuracy:** 73.90%
* **At-Risk Detection (Recall for 'FAIL'):** 78.43%
* **ROC-AUC Score:** 0.8386
* **Key Finding:** Permutation feature analysis identified *Project Scores* (0.1860) as the primary academic driver, while *Stress Level* served as the critical behavioral "tie-breaker" for borderline students.

---

## ⚙️ How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/eydditt/student-performance-prediction-using-knn-algorithm.git](https://github.com/eydditt/student-performance-prediction-using-knn-algorithm.git)
   cd student-performance-prediction-using-knn-algorithm

   pip install pandas numpy scikit-learn imbalanced-learn flask matplotlib seaborn

   python student_performance_app.py
