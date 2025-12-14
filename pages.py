# =========================================================
# Page: Upload Data
# Purpose: Load dataset and store it for next steps
# =========================================================

# -------------------------
# Imports
# -------------------------
import streamlit as st
import pandas as pd


# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Upload Data",
    layout="wide"
)

st.title("📤 رفع البيانات")
st.write("ارفع ملف البيانات (CSV أو Excel) لبدء مشروع تحليل البيانات")


# -------------------------
# Helper Functions
# -------------------------
def load_data(file):
    """Load CSV or Excel file into DataFrame."""
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        return None


def init_session_state():
    """Initialize session state variables."""
    if "df" not in st.session_state:
        st.session_state["df"] = None


# -------------------------
# Initialize Session State
# -------------------------
init_session_state()


# -------------------------
# Main Logic - File Upload
# -------------------------
uploaded_file = st.file_uploader(
    "اختر ملف البيانات",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file)
        st.session_state["df"] = df

        st.success("تم رفع البيانات بنجاح ✅")
        st.write("معاينة البيانات:")
        st.dataframe(df.head())

        st.caption(f"عدد الصفوف: {df.shape[0]} | عدد الأعمدة: {df.shape[1]}")

    except Exception as e:
        st.error(f"حدث خطأ أثناء تحميل البيانات: {e}")

else:
    st.info("لم يتم رفع أي بيانات بعد.")


# -------------------------
# Custom Code Section
# -------------------------
st.divider()
st.subheader("🧩 كود مخصص (اختياري)")
st.write(
    """
    يمكنك كتابة كود Python للتعامل مع البيانات.
    يجب أن يكون اسم البيانات **df**.
    
    مثال:
    ```python
    df = df.dropna()
    df["new_col"] = df["old_col"] * 2
    ```
    """
)

custom_code = st.text_area(
    "اكتب كودك هنا:",
    height=220
)

run_custom_code = st.button("تشغيل الكود")

if run_custom_code:
    if st.session_state["df"] is None:
        st.warning("يجب رفع البيانات أولاً.")
    else:
        try:
            df = st.session_state["df"]

            # Execute user code safely in local scope
            local_scope = {"df": df}
            exec(custom_code, {}, local_scope)

            # Update df if modified
            if "df" in local_scope:
                st.session_state["df"] = local_scope["df"]
                st.success("تم تنفيذ الكود بنجاح ✅")
                st.dataframe(st.session_state["df"].head())
            else:
                st.warning("لم يتم تعديل df داخل الكود.")

        except Exception as e:
            st.error(f"خطأ أثناء تنفيذ الكود: {e}")