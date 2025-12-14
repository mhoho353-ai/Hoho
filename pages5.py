# =========================================================
# Page: Feature Selection
# Purpose: Select important features for modeling
# =========================================================

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    RFE
)
from sklearn.linear_model import LogisticRegression, LinearRegression


# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Feature Selection",
    layout="wide"
)

st.title("🎯 اختيار السمات (Feature Selection)")
st.write("تقليل عدد المتغيرات لتحسين أداء النموذج")


# -------------------------
# Helper Functions
# -------------------------
def data_exists():
    return "df" in st.session_state and st.session_state["df"] is not None


def split_features_target(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def variance_threshold_selection(X, threshold):
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()]
    return pd.DataFrame(X_selected, columns=selected_cols), selected_cols.tolist()


def select_k_best(X, y, k, problem_type):
    if problem_type == "Classification":
        selector = SelectKBest(score_func=f_classif, k=k)
    else:
        selector = SelectKBest(score_func=f_regression, k=k)

    X_selected = selector.fit_transform(X, y)
    selected_cols = X.columns[selector.get_support()]
    return pd.DataFrame(X_selected, columns=selected_cols), selected_cols.tolist()


def rfe_selection(X, y, n_features, problem_type):
    if problem_type == "Classification":
        model = LogisticRegression(max_iter=1000)
    else:
        model = LinearRegression()

    selector = RFE(model, n_features_to_select=n_features)
    selector.fit(X, y)

    selected_cols = X.columns[selector.support_]
    X_selected = X[selected_cols]
    return X_selected, selected_cols.tolist()


# -------------------------
# Main Logic
# -------------------------
if not data_exists():
    st.warning("⚠️ لا توجد بيانات. انتقل إلى صفحة رفع البيانات.")
    st.stop()

df = st.session_state["df"]

st.subheader("⚙️ الإعدادات الأساسية")

target_column = st.selectbox(
    "اختر عمود الهدف (Target)",
    ["لا شيء"] + df.columns.tolist()
)

if target_column == "لا شيء":
    st.info("يرجى اختيار عمود الهدف للمتابعة.")
    st.stop()


problem_type = st.selectbox(
    "نوع المشكلة",
    ["Classification", "Regression"]
)

# -------------------------
# Prepare X and y
# -------------------------
X, y = split_features_target(df, target_column)

# نأخذ الأعمدة الرقمية فقط
X = X.select_dtypes(include=np.number)

if X.empty:
    st.error("⚠️ لا توجد أعمدة رقمية لاختيار السمات.")
    st.stop()


# -------------------------
# Feature Selection Method
# -------------------------
st.subheader("🧪 طريقة اختيار السمات")

method = st.selectbox(
    "اختر الطريقة",
    ["لا شيء", "Variance Threshold", "SelectKBest", "RFE"]
)

selected_features = X.columns.tolist()

if method == "Variance Threshold":
    threshold = st.slider("قيمة التباين", 0.0, 1.0, 0.0)
    X, selected_features = variance_threshold_selection(X, threshold)

elif method == "SelectKBest":
    k = st.slider("عدد السمات المختارة (k)", 1, X.shape[1], min(5, X.shape[1]))
    X, selected_features = select_k_best(X, y, k, problem_type)

elif method == "RFE":
    n_features = st.slider(
        "عدد السمات المختارة",
        1,
        X.shape[1],
        min(5, X.shape[1])
    )
    X, selected_features = rfe_selection(X, y, n_features, problem_type)

else:
    st.info("لم يتم اختيار أي طريقة — لم يتم تغيير السمات.")


# -------------------------
# Save Results
# -------------------------
st.divider()
if st.button("💾 حفظ السمات المختارة"):
    st.session_state["X"] = X
    st.session_state["y"] = y
    st.session_state["selected_features"] = selected_features

    st.success("تم حفظ السمات بنجاح ✅")
    st.write("السمات المختارة:")
    st.write(selected_features)


# -------------------------
# Custom External Code Section (REQUIRED)
# -------------------------
st.divider()
st.subheader("🧩 مربع إضافة الكود الخارجي (اختياري)")
st.write(
    """
    ✔ موجود في **كل الصفحات**
    ✔ يمكنك التعديل على X و y
    ✔ مناسب لتجارب مخصصة
    
    مثال:
    ```python
    X = X.drop(columns=["unwanted_feature"])
    ```
    """
)

external_code = st.text_area(
    "اكتب الكود الخارجي هنا:",
    height=220
)

run_external_code = st.button("تشغيل الكود الخارجي")

if run_external_code:
    try:
        local_scope = {"X": X, "y": y, "pd": pd, "np": np}
        exec(external_code, {}, local_scope)

        if "X" in local_scope:
            X = local_scope["X"]
            st.session_state["X"] = X

        if "y" in local_scope:
            y = local_scope["y"]
            st.session_state["y"] = y

        st.success("تم تنفيذ الكود الخارجي بنجاح ✅")
        st.write("أبعاد X بعد التعديل:", X.shape)

    except Exception as e:
        st.error(f"خطأ في الكود الخارجي: {e}")