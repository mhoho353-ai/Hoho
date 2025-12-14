# =========================================================
# Page: Models Training & AutoML
# Purpose: Train ML models and optional AutoML
# =========================================================

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, r2_score


# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Models & AutoML",
    layout="wide"
)

st.title("🤖 اختيار النماذج وتدريبها")
st.write("تدريب النماذج التقليدية أو استخدام AutoML")


# -------------------------
# Helper Functions
# -------------------------
def data_ready():
    keys = ["X_train", "X_test", "y_train", "y_test"]
    return all(k in st.session_state for k in keys)


def get_problem_type(y):
    """Infer problem type from target."""
    if y.nunique() <= 20:
        return "Classification"
    return "Regression"


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


# -------------------------
# Check Data
# -------------------------
if not data_ready():
    st.warning("⚠️ لم يتم العثور على بيانات التدريب. انتقل أولاً لصفحة التقسيم.")
    st.stop()

X_train = st.session_state["X_train"]
X_test = st.session_state["X_test"]
y_train = st.session_state["y_train"]
y_test = st.session_state["y_test"]

problem_type = get_problem_type(y_train)

st.info(f"📌 نوع المشكلة المكتشف تلقائيًا: **{problem_type}**")


# -------------------------
# Model Selection
# -------------------------
st.subheader("⚙️ اختيار النموذج")

if problem_type == "Classification":
    model_name = st.selectbox(
        "اختر نموذج التصنيف",
        ["Logistic Regression", "Random Forest", "SVM"]
    )
else:
    model_name = st.selectbox(
        "اختر نموذج الانحدار",
        ["Linear Regression", "Ridge", "Random Forest"]
    )


# -------------------------
# Model Initialization
# -------------------------
if problem_type == "Classification":
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        model = SVC(probability=True)
else:
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge(alpha=1.0)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42)


# -------------------------
# Train Model
# -------------------------
if st.button("🚀 تدريب النموذج"):
    try:
        model = train_model(model, X_train, y_train)
        st.session_state["model"] = model
        st.session_state["problem_type"] = problem_type

        st.success("تم تدريب النموذج بنجاح ✅")

    except Exception as e:
        st.error(f"خطأ أثناء التدريب: {e}")


# -------------------------
# AutoML Section (OPTIONAL)
# -------------------------
st.divider()
st.subheader("🧠 AutoML (اختياري)")

use_automl = st.checkbox("استخدام AutoML (PyCaret إن توفر)")

if use_automl:
    try:
        from pycaret.classification import setup as cls_setup, compare_models as cls_compare
        from pycaret.regression import setup as reg_setup, compare_models as reg_compare

        if problem_type == "Classification":
            data = pd.concat([X_train, y_train], axis=1)
            cls_setup(data=data, target=y_train.name, silent=True, html=False)
            best_model = cls_compare()
        else:
            data = pd.concat([X_train, y_train], axis=1)
            reg_setup(data=data, target=y_train.name, silent=True, html=False)
            best_model = reg_compare()

        st.session_state["model"] = best_model
        st.success("تم اختيار أفضل نموذج باستخدام AutoML ✅")

    except Exception as e:
        st.warning("⚠️ AutoML غير متوفر في البيئة الحالية.")
        st.caption(str(e))


# -------------------------
# Save Model
# -------------------------
st.divider()
if "model" in st.session_state:
    st.success("📦 النموذج الحالي جاهز للاستخدام في التقييم")


# -------------------------
# Custom External Code Section (MANDATORY)
# -------------------------
st.divider()
st.subheader("🧩 مربع إضافة الكود الخارجي (اختياري)")
st.write(
    """
    ✔ موجود في كل الصفحات  
    ✔ يمكنك تعديل النموذج أو إنشاء نموذجك الخاص  
    ✔ المتغيرات المتاحة: model, X_train, y_train
    
    مثال:
    ```python
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train, y_train)
    ```
    """
)

external_code = st.text_area(
    "اكتب الكود الخارجي هنا:",
    height=240
)

run_external_code = st.button("تشغيل الكود الخارجي")

if run_external_code:
    try:
        local_scope = {
            "model": st.session_state.get("model"),
            "X_train": X_train,
            "y_train": y_train
        }

        exec(external_code, {}, local_scope)

        if "model" in local_scope:
            st.session_state["model"] = local_scope["model"]
            st.success("تم تنفيذ الكود الخارجي وتحديث النموذج ✅")

    except Exception as e:
        st.error(f"خطأ في الكود الخارجي: {e}")