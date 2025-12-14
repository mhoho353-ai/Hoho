# =========================================================
# Page: Pipeline & GridSearch
# Purpose: Hyperparameter tuning with Pipeline
# =========================================================

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Pipeline & GridSearch",
    layout="wide"
)

st.title("🧠 Pipeline + GridSearch")
st.write("تحسين أداء النموذج باستخدام Cross-Validation")


# -------------------------
# Helper
# -------------------------
def data_ready():
    keys = ["X_train", "y_train"]
    return all(k in st.session_state for k in keys)


def infer_problem_type(y):
    return "Classification" if y.nunique() <= 20 else "Regression"


# -------------------------
# Check Data
# -------------------------
if not data_ready():
    st.warning("⚠️ بيانات التدريب غير متوفرة.")
    st.stop()

X_train = st.session_state["X_train"]
y_train = st.session_state["y_train"]

problem_type = infer_problem_type(y_train)
st.info(f"📌 نوع المشكلة: {problem_type}")


# -------------------------
# Model Selection
# -------------------------
st.subheader("⚙️ اختيار النموذج")

if problem_type == "Classification":
    model_name = st.selectbox(
        "النموذج",
        ["Logistic Regression", "Random Forest", "SVM"]
    )
else:
    model_name = st.selectbox(
        "النموذج",
        ["Ridge", "Random Forest", "SVR"]
    )


# -------------------------
# Build Pipeline & Params
# -------------------------
if problem_type == "Classification":

    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
        param_grid = {
            "model__C": [0.01, 0.1, 1, 10]
        }

    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20]
        }

    else:
        model = SVC(probability=True)
        param_grid = {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"]
        }

else:

    if model_name == "Ridge":
        model = Ridge()
        param_grid = {
            "model__alpha": [0.1, 1, 10]
        }

    elif model_name == "Random Forest":
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10]
        }

    else:
        model = SVR()
        param_grid = {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"]
        }


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", model)
])


# -------------------------
# GridSearch
# -------------------------
if st.button("🚀 تشغيل GridSearchCV"):
    try:
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring="accuracy" if problem_type == "Classification" else "r2",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        st.session_state["model"] = grid.best_estimator_
        st.session_state["best_params"] = grid.best_params_
        st.session_state["cv_score"] = grid.best_score_

        st.success("تم العثور على أفضل نموذج ✅")
        st.write("أفضل المعاملات:")
        st.json(grid.best_params_)
        st.write("أفضل نتيجة CV:", round(grid.best_score_, 4))

    except Exception as e:
        st.error(f"خطأ أثناء GridSearch: {e}")


# -------------------------
# Custom External Code
# -------------------------
st.divider()
st.subheader("🧩 مربع إضافة الكود الخارجي")
st.write("يمكنك تعديل الـ pipeline أو إنشاء GridSearch مخصص")

external_code = st.text_area("الكود الخارجي:", height=220)

if st.button("تشغيل الكود الخارجي"):
    try:
        local_scope = {
            "pipeline": pipeline,
            "X_train": X_train,
            "y_train": y_train
        }
        exec(external_code, {}, local_scope)
        st.success("تم تنفيذ الكود الخارجي ✅")
    except Exception as e:
        st.error(e)