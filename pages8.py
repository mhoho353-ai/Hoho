# =========================================================
# Page: Model Evaluation & Final Code Export
# Purpose: Evaluate trained model and export clean code
# =========================================================

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Model Evaluation & Final Code",
    layout="wide"
)

st.title("📊 تقييم النماذج واستخراج الكود النهائي")


# -------------------------
# Helper Functions
# -------------------------
def requirements_exist():
    keys = ["model", "X_test", "y_test"]
    return all(k in st.session_state for k in keys)


def detect_problem_type(y):
    return "Classification" if y.nunique() <= 20 else "Regression"


def generate_final_code(problem_type, model_name):
    """
    Generate clean, understandable Python code
    suitable for Kaggle or production.
    """
    code = f'''
# ==============================
# Final Machine Learning Pipeline
# ==============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Load data
df = pd.read_csv("data.csv")  # change path if needed

# 2. Select target
TARGET = "target_column"  # <-- change this

X = df.drop(columns=[TARGET])
y = df[TARGET]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 4. Model
model = {model_name}

# 5. Train
model.fit(X_train, y_train)

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluation
'''

    if problem_type == "Classification":
        code += '''
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="weighted"))
print("Recall:", recall_score(y_test, y_pred, average="weighted"))
print("F1:", f1_score(y_test, y_pred, average="weighted"))
'''
    else:
        code += '''
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))
'''

    return code


# -------------------------
# Check Requirements
# -------------------------
if not requirements_exist():
    st.warning("⚠️ لا يوجد نموذج أو بيانات اختبار. تأكد من تدريب نموذج أولاً.")
    st.stop()

model = st.session_state["model"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

problem_type = detect_problem_type(y_test)

st.info(f"📌 نوع المشكلة: **{problem_type}**")


# -------------------------
# Predictions
# -------------------------
y_pred = model.predict(X_test)


# -------------------------
# Evaluation Metrics
# -------------------------
st.subheader("📈 المقاييس")

if problem_type == "Classification":
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", round(acc, 4))
    col2.metric("Precision", round(prec, 4))
    col3.metric("Recall", round(rec, 4))
    col4.metric("F1-score", round(f1, 4))

else:
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", round(mae, 4))
    col2.metric("MSE", round(mse, 4))
    col3.metric("R2", round(r2, 4))


# -------------------------
# Visualizations
# -------------------------
st.subheader("📊 الرسوم")

if problem_type == "Classification":
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    st.pyplot(fig)

    if hasattr(model, "predict_proba"):
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
        st.pyplot(fig)


# -------------------------
# Final Code Export
# -------------------------
st.divider()
st.subheader("📦 تحميل الكود النهائي")

model_name = model.__class__.__name__ + "()"
final_code = generate_final_code(problem_type, model_name)

st.code(final_code, language="python")

st.download_button(
    label="⬇️ تحميل الكود النهائي",
    data=final_code,
    file_name="final_ml_pipeline.py",
    mime="text/plain"
)