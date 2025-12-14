# =========================================================
# Page: Train / Test Split
# Purpose: Split data into training and testing sets
# =========================================================

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Split Data",
    layout="wide"
)

st.title("✂️ تقسيم البيانات (Train / Test Split)")
st.write("تقسيم البيانات خطوة أساسية قبل تدريب النماذج")


# -------------------------
# Helper Functions
# -------------------------
def get_X_y():
    """
    Priority:
    1) Use X, y if available (from Feature Selection)
    2) Otherwise extract from df using selected target
    """
    if "X" in st.session_state and "y" in st.session_state:
        return st.session_state["X"], st.session_state["y"]

    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]
        target = st.session_state.get("target_column", None)

        if target and target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]
            X = X.select_dtypes(include=np.number)
            return X, y

    return None, None


def data_ready(X, y):
    return X is not None and y is not None and not X.empty


# -------------------------
# Load Data
# -------------------------
X, y = get_X_y()

if not data_ready(X, y):
    st.warning("⚠️ لا توجد بيانات جاهزة للتقسيم. تأكد من اختيار Target والسمات.")
    st.stop()


# -------------------------
# Split Settings
# -------------------------
st.subheader("⚙️ إعدادات التقسيم")

test_size = st.slider(
    "نسبة بيانات الاختبار (Test Size)",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05
)

shuffle = st.checkbox("Shuffle البيانات", value=True)

random_state = st.number_input(
    "Random State (لإعادة نفس النتائج)",
    min_value=0,
    max_value=9999,
    value=42
)


# -------------------------
# Apply Split
# -------------------------
if st.button("✂️ تنفيذ التقسيم"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            shuffle=shuffle,
            random_state=random_state if shuffle else None
        )

        # Save to session state
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

        st.success("تم تقسيم البيانات بنجاح ✅")

        col1, col2 = st.columns(2)
        with col1:
            st.write("🔵 Train Shape")
            st.write(X_train.shape)
        with col2:
            st.write("🟠 Test Shape")
            st.write(X_test.shape)

    except Exception as e:
        st.error(f"حدث خطأ أثناء التقسيم: {e}")


# -------------------------
# Custom External Code Section (REQUIRED)
# -------------------------
st.divider()
st.subheader("🧩 مربع إضافة الكود الخارجي (اختياري)")
st.write(
    """
    ✔ موجود في جميع الصفحات  
    ✔ يمكنك التعديل على X_train, X_test, y_train, y_test  
    
    مثال:
    ```python
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
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
        local_scope = {
            "X_train": st.session_state.get("X_train"),
            "X_test": st.session_state.get("X_test"),
            "y_train": st.session_state.get("y_train"),
            "y_test": st.session_state.get("y_test"),
            "pd": pd,
            "np": np
        }

        exec(external_code, {}, local_scope)

        # Update session state if modified
        for key in ["X_train", "X_test", "y_train", "y_test"]:
            if key in local_scope and local_scope[key] is not None:
                st.session_state[key] = local_scope[key]

        st.success("تم تنفيذ الكود الخارجي بنجاح ✅")

    except Exception as e:
        st.error(f"خطأ في الكود الخارجي: {e}")