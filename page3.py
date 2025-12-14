# =========================================================
# Page: Data Cleaning
# Purpose: Handle missing values, outliers, encoding, types
# =========================================================

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd
import numpy as np


# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Clean Data",
    layout="wide"
)

st.title("🧹 تنظيف البيانات")
st.write("تنظيف البيانات خطوة أساسية قبل النمذجة")


# -------------------------
# Helper Functions
# -------------------------
def data_exists():
    """Check if data exists."""
    return "df" in st.session_state and st.session_state["df"] is not None


def handle_missing(df, strategy, columns):
    """Handle missing values."""
    if not columns:
        return df

    df = df.copy()

    for col in columns:
        if strategy == "حذف الصفوف":
            df = df.dropna(subset=[col])
        elif strategy == "تعويض بالمتوسط":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "تعويض بالوسيط":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "تعويض بالقيمة الأكثر تكرارًا":
            df[col] = df[col].fillna(df[col].mode()[0])

    return df


def remove_outliers_iqr(df, columns):
    """Remove outliers using IQR."""
    df = df.copy()

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def encode_categorical(df, columns):
    """Encode categorical columns using one-hot encoding."""
    if not columns:
        return df
    return pd.get_dummies(df, columns=columns, drop_first=True)


def change_column_type(df, column, new_type):
    """Change data type of a column."""
    df = df.copy()
    try:
        df[column] = df[column].astype(new_type)
    except:
        pass
    return df


# -------------------------
# Main Logic
# -------------------------
if not data_exists():
    st.warning("⚠️ لم يتم العثور على بيانات. انتقل إلى صفحة رفع البيانات أولاً.")
    st.stop()

df_original = st.session_state["df"]
df = df_original.copy()


# -------------------------
# Missing Values Section
# -------------------------
st.subheader("❗ التعامل مع القيم المفقودة")

missing_cols = df.columns[df.isnull().any()].tolist()

if missing_cols:
    strategy = st.selectbox(
        "اختر طريقة التعامل",
        ["لا شيء", "حذف الصفوف", "تعويض بالمتوسط", "تعويض بالوسيط", "تعويض بالقيمة الأكثر تكرارًا"]
    )

    selected_cols = st.multiselect(
        "اختر الأعمدة",
        missing_cols
    )

    if strategy != "لا شيء":
        df = handle_missing(df, strategy, selected_cols)
else:
    st.success("لا توجد قيم مفقودة ✅")


# -------------------------
# Outliers Section
# -------------------------
st.subheader("🚨 القيم الشاذة (Outliers)")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

outlier_cols = st.multiselect(
    "اختر الأعمدة الرقمية لمعالجة القيم الشاذة (IQR)",
    numeric_cols
)

if outlier_cols:
    df = remove_outliers_iqr(df, outlier_cols)


# -------------------------
# Encoding Section
# -------------------------
st.subheader("🔤 ترميز المتغيرات النصية")

cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

encode_cols = st.multiselect(
    "اختر الأعمدة النصية للترميز",
    cat_cols
)

if encode_cols:
    df = encode_categorical(df, encode_cols)


# -------------------------
# Change Data Type Section
# -------------------------
st.subheader("🔄 تغيير نوع البيانات")

col_to_change = st.selectbox(
    "اختر العمود",
    ["لا شيء"] + df.columns.tolist()
)

if col_to_change != "لا شيء":
    new_type = st.selectbox(
        "اختر النوع الجديد",
        ["int", "float", "str"]
    )
    df = change_column_type(df, col_to_change, new_type)


# -------------------------
# Save Cleaned Data
# -------------------------
st.divider()
if st.button("💾 حفظ البيانات بعد التنظيف"):
    st.session_state["df"] = df
    st.success("تم حفظ البيانات المنظفة بنجاح ✅")
    st.dataframe(df.head())


# -------------------------
# Custom Code Section (IMPORTANT)
# -------------------------
st.divider()
st.subheader("🧩 مربع إضافة الكود الخارجي (اختياري)")
st.write(
    """
    ✔ هذا المربع موجود في **كل الصفحات**  
    ✔ يمكنك كتابة أي كود Python  
    ✔ المتغير الأساسي هو **df**
    
    مثال:
    ```python
    df = df.drop(columns=["id"])
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
        local_scope = {"df": df}
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