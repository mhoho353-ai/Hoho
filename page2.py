# =========================================================
# Page: Explore Data (EDA)
# Purpose: Data exploration and understanding
# =========================================================

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Explore Data",
    layout="wide"
)

st.title("📊 استكشاف البيانات (EDA)")
st.write("فهم البيانات قبل التنظيف والنمذجة خطوة أساسية")


# -------------------------
# Helper Functions
# -------------------------
def check_data_exists():
    """Check if dataset exists in session state."""
    return "df" in st.session_state and st.session_state["df"] is not None


def basic_info(df):
    """Return basic dataset information."""
    info = {
        "عدد الصفوف": df.shape[0],
        "عدد الأعمدة": df.shape[1],
        "الأعمدة الرقمية": df.select_dtypes(include=np.number).shape[1],
        "الأعمدة النصية": df.select_dtypes(exclude=np.number).shape[1],
        "القيم المفقودة": df.isnull().sum().sum()
    }
    return pd.DataFrame(info, index=["القيم"])


def missing_values_table(df):
    """Missing values summary."""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    return missing.sort_values(ascending=False)


# -------------------------
# Main Logic
# -------------------------
if not check_data_exists():
    st.warning("⚠️ لم يتم العثور على بيانات. انتقل إلى صفحة رفع البيانات أولاً.")
    st.stop()

df = st.session_state["df"]


# -------------------------
# Dataset Overview
# -------------------------
st.subheader("🔍 نظرة عامة على البيانات")
st.dataframe(basic_info(df))


# -------------------------
# Data Types
# -------------------------
st.subheader("🧬 أنواع البيانات")
dtypes_df = pd.DataFrame(df.dtypes, columns=["نوع البيانات"])
st.dataframe(dtypes_df)


# -------------------------
# Missing Values
# -------------------------
st.subheader("❗ القيم المفقودة")
missing_df = missing_values_table(df)

if missing_df.empty:
    st.success("لا توجد قيم مفقودة ✅")
else:
    st.dataframe(missing_df)


# -------------------------
# Descriptive Statistics
# -------------------------
st.subheader("📈 الإحصاءات الوصفية")
st.dataframe(df.describe(include="all").transpose())


# -------------------------
# Unique Values
# -------------------------
st.subheader("🔢 عدد القيم الفريدة لكل عمود")
unique_values = df.nunique()
st.dataframe(unique_values.to_frame("عدد القيم الفريدة"))


# -------------------------
# Visualizations
# -------------------------
st.subheader("📉 الرسوم الاستكشافية")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if numeric_cols:
    selected_col = st.selectbox("اختر عمودًا رقميًا للرسم", numeric_cols)

    col1, col2 = st.columns(2)

    with col1:
        st.write("Histogram")
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Boxplot")
        fig, ax = plt.subplots()
        sns.boxplot(y=df[selected_col], ax=ax)
        st.pyplot(fig)
else:
    st.info("لا توجد أعمدة رقمية للرسم.")


# -------------------------
# Correlation Matrix
# -------------------------
st.subheader("🔗 الارتباط بين المتغيرات")

if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
else:
    st.info("عدد الأعمدة الرقمية غير كافٍ لحساب الارتباط.")


# -------------------------
# Custom Code Section
# -------------------------
st.divider()
st.subheader("🧩 كود مخصص (اختياري)")
st.write(
    """
    يمكنك إضافة أي كود استكشافي إضافي.
    المتغير المتاح هو **df**.
    
    مثال:
    ```python
    df.groupby("target").mean()
    ```
    """
)

custom_code = st.text_area(
    "اكتب كودك هنا:",
    height=200
)

run_custom_code = st.button("تشغيل الكود")

if run_custom_code:
    try:
        local_scope = {"df": df, "pd": pd, "np": np}
        exec(custom_code, {}, local_scope)
        st.success("تم تنفيذ الكود بنجاح ✅")
    except Exception as e:
        st.error(f"خطأ في الكود: {e}")