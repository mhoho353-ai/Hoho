# =========================================================
# Page: Scaling Data
# Purpose: Apply feature scaling to numeric columns
# =========================================================

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)


# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Scaling Data",
    layout="wide"
)

st.title("📐 تدريج البيانات (Scaling)")
st.write("توحيد نطاق القيم الرقمية لتحسين أداء النماذج")


# -------------------------
# Helper Functions
# -------------------------
def data_exists():
    return "df" in st.session_state and st.session_state["df"] is not None


def get_scaler(name):
    """Return scaler object based on selection."""
    if name == "StandardScaler":
        return StandardScaler()
    if name == "MinMaxScaler":
        return MinMaxScaler()
    if name == "RobustScaler":
        return RobustScaler()
    return None


def apply_scaling(df, columns, scaler):
    """Apply scaling to selected columns only."""
    df = df.copy()
    df[columns] = scaler.fit_transform(df[columns])
    return df


# -------------------------
# Main Logic
# -------------------------
if not data_exists():
    st.warning("⚠️ لا توجد بيانات. انتقل أولاً إلى صفحة رفع البيانات.")
    st.stop()

df_original = st.session_state["df"]
df = df_original.copy()


# -------------------------
# Select Numeric Columns
# -------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    st.info("لا توجد أعمدة رقمية قابلة للتدريج.")
else:
    st.subheader("⚙️ إعدادات التدريج")

    scaler_name = st.selectbox(
        "اختر نوع الـ Scaler",
        ["لا شيء", "StandardScaler", "MinMaxScaler", "RobustScaler"]
    )

    selected_columns = st.multiselect(
        "اختر الأعمدة الرقمية",
        numeric_cols
    )

    if scaler_name != "لا شيء" and selected_columns:
        scaler = get_scaler(scaler_name)
        df = apply_scaling(df, selected_columns, scaler)
        st.success("تم تطبيق التدريج بنجاح ✅")
    else:
        st.info("لم يتم اختيار Scaler أو أعمدة — لم يتم تعديل البيانات.")


# -------------------------
# Save Scaled Data
# -------------------------
st.divider()
if st.button("💾 حفظ البيانات بعد التدريج"):
    st.session_state["df"] = df
    st.success("تم حفظ البيانات بعد التدريج ✅")
    st.dataframe(df.head())


# -------------------------
# Custom External Code Section (FIXED & REQUIRED)
# -------------------------
st.divider()
st.subheader("🧩 مربع إضافة الكود الخارجي (اختياري)")
st.write(
    """
    ✔ هذا المربع موجود في **جميع الصفحات**
    ✔ يمكنك كتابة أي كود Python
    ✔ المتغير المستخدم هو **df**

    مثال:
    ```python
    df["log_income"] = np.log1p(df["income"])
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
        local_scope = {"df": df, "np": np, "pd": pd}
        exec(external_code, {}, local_scope)

        if "df" in local_scope:
            df = local_scope["df"]
            st.session_state["df"] = df
            st.success("تم تنفيذ الكود الخارجي وحفظ النتائج ✅")
            st.dataframe(df.head())
        else:
            st.warning("لم يتم تعديل df داخل الكود.")

    except Exception as e:
        st.error(f"خطأ في الكود الخارجي: {e}")