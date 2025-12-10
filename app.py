import streamlit as st
import pandas as pd
import io

st.set_page_config(page_title="تحليل البيانات", layout="wide")

# تفعيل التخزين
if "df" not in st.session_state:
    st.session_state.df = None
if "df_original" not in st.session_state:
    st.session_state.df_original = None

# ---------------------- واجهة الصفحة ----------------------
st.title("📊 تحليل البيانات")

st.subheader("📁 رفع البيانات")

uploaded_file = st.file_uploader("ارفع ملف CSV أو Excel", type=["csv", "xlsx"])

# ---------------------- قراءة البيانات ----------------------
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df = df.copy()        # نسخة عمل
        st.session_state.df_original = df.copy()  # نسخة احتياطية

        st.success("✔ تم رفع البيانات بنجاح!")
    except:
        st.error("⚠ حدث خطأ أثناء قراءة الملف")

# ---------------------- عرض جزء من البيانات ----------------------
if st.session_state.df is not None:
    st.subheader("🔍 عرض أول 5 صفوف من البيانات")
    st.dataframe(st.session_state.df.head())

    st.write("---")
    st.subheader("🧭 أدوات استكشاف البيانات")

    col1, col2, col3 = st.columns(3)

    # ---------------------- أزرار الاستكشاف ----------------------
    with col1:
        if st.button("عرض أول 5 صفوف"):
            st.dataframe(st.session_state.df.head())

        if st.button("شكل البيانات (shape)"):
            st.write(st.session_state.df.shape)

        if st.button("أنواع البيانات"):
            st.write(st.session_state.df.dtypes)

    with col2:
        if st.button("عرض آخر 5 صفوف"):
            st.dataframe(st.session_state.df.tail())

        if st.button("أسماء الأعمدة"):
            st.write(list(st.session_state.df.columns))

        if st.button("معلومات البيانات (info)"):
            buffer = io.StringIO()
            st.session_state.df.info(buf=buffer)
            info_text = buffer.getvalue()
            st.text(info_text)

    with col3:
        if st.button("الإحصائيات الوصفية (describe)"):
            st.write(st.session_state.df.describe())

    st.write("---")

    # ---------------------- مربع دوال خاصة ----------------------
    st.subheader("✏️ اكتب دالة استكشاف خاصة")

    code_input = st.text_area("اكتب أي كود مثل: df.isnull().sum()")

    if st.button("تشغيل الدالة"):
        try:
            result = eval(code_input, {"df": st.session_state.df})
            st.write(result)
        except Exception as e:
            st.error(f"⚠ خطأ في تنفيذ الكود: {e}")

    st.write("---")

    # ---------------------- زر الصفحة التالية ----------------------
    if st.button("➡ الصفحة التالية: تنظيف البيانات"):
        st.switch_page("page_cleaning.py")
