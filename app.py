# app.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

st.set_page_config(page_title="Simple DataLab", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Helpers & Caching
# ---------------------------

@st.cache_data(show_spinner=False)
def load_preview(file_buf, filetype):
    """Return a small preview and inferred dtypes without storing big data multiple times."""
    if filetype == "csv":
        df = pd.read_csv(file_buf, nrows=1000)
    elif filetype == "excel":
        df = pd.read_excel(file_buf, engine="openpyxl", nrows=1000)
    elif filetype == "json":
        df = pd.read_json(file_buf, lines=False)
    else:
        df = pd.read_csv(file_buf, nrows=1000)
    return df

@st.cache_data(show_spinner=False)
def read_full(file_buf, filetype, sheet_name=None):
    if filetype == "csv":
        return pd.read_csv(file_buf)
    elif filetype == "excel":
        return pd.read_excel(file_buf, sheet_name=sheet_name, engine="openpyxl")
    elif filetype == "json":
        return pd.read_json(file_buf, lines=False)
    else:
        return pd.read_csv(file_buf)

def to_excel_bytes(df: pd.DataFrame):
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

def df_info_summary(df: pd.DataFrame):
    info = {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "memory_mb": df.memory_usage(deep=True).sum() / (1024**2),
        "columns": []
    }
    for c in df.columns:
        series = df[c]
        info["columns"].append({
            "name": c,
            "dtype": str(series.dtype),
            "non_null": int(series.count()),
            "null_pct": float((series.isna().sum() / len(series)) * 100),
            "n_unique": int(series.nunique(dropna=True)),
            "example": str(series.dropna().iloc[0]) if series.dropna().shape[0] > 0 else ""
        })
    return info

def detect_column_type(series: pd.Series):
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_categorical_dtype(series):
        return "category"
    return "text"

def simple_outlier_mask(series: pd.Series, method="iqr", factor=1.5):
    if method == "iqr":
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        low = q1 - factor * iqr
        high = q3 + factor * iqr
        return (series < low) | (series > high)
    else:
        return pd.Series(False, index=series.index)

# ---------------------------
# Session state initialization
# ---------------------------
if "df" not in st.session_state:
    st.session_state.df = None
if "original_df" not in st.session_state:
    st.session_state.original_df = None
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "filetype" not in st.session_state:
    st.session_state.filetype = None
if "sheet" not in st.session_state:
    st.session_state.sheet = None

# ---------------------------
# UI: Sidebar Navigation
# ---------------------------
st.sidebar.title("Simple DataLab")
page = st.sidebar.radio("الخطوات", ("البدء", "رفع البيانات", "تنظيف", "تحليل", "نماذج", "رسم", "تصدير"))

# ---------------------------
# Page: Start
# ---------------------------
if page == "البدء":
    st.title("مرحبًا في Simple DataLab")
    st.markdown(
        """
        تطبيق بسيط لتحليل البيانات — رفع، تنظيف، تحليل، تدريب نماذج، رسم وتصدير.
        الهدف: واجهة سهلة وبسيطة تتيح لك التحكم الكامل بالبيانات.
        """
    )
    st.info("ابدأ من الشريط الجانبي: اختر 'رفع البيانات' لرفع ملفك الأول (CSV, Excel, JSON).")

# ---------------------------
# Page: Upload
# ---------------------------
if page == "رفع البيانات":
    st.header("رفع البيانات")
    st.write("اسحب أو اختر ملف CSV / Excel / JSON. (يدعم ملفات حتى عدة ميجابايت على Streamlit Cloud).")

    uploaded_file = st.file_uploader("اسحب ملف هنا", type=["csv", "xlsx", "xls", "json"], accept_multiple_files=False)
    if uploaded_file is not None:
        # detect type
        fname = uploaded_file.name.lower()
        if fname.endswith(".csv"):
            ftype = "csv"
        elif fname.endswith(".xlsx") or fname.endswith(".xls"):
            ftype = "excel"
        elif fname.endswith(".json"):
            ftype = "json"
        else:
            ftype = "csv"

        st.session_state.uploaded_file = uploaded_file
        st.session_state.filetype = ftype

        if ftype == "excel":
            # list sheets
            try:
                e = pd.ExcelFile(uploaded_file, engine="openpyxl")
                sheets = e.sheet_names
                sheet = st.selectbox("اختر الورقة (sheet) إن كانت متعددة", sheets)
                st.session_state.sheet = sheet
                uploaded_file.seek(0)
                # preview
                preview = read_full(uploaded_file, "excel", sheet_name=sheet).head(500)
                st.write("معاينة أولية (أول 500 صف أو أقل)")
                st.dataframe(preview.sample(min(len(preview), 10)))
            except Exception as e:
                st.error("خطأ في قراءة ملف Excel: " + str(e))
        else:
            # preview generic
            uploaded_file.seek(0)
            preview = load_preview(uploaded_file, ftype)
            st.write("معاينة سريعة (حتى 1000 صف)")
            st.dataframe(preview.head(10))
            st.write(preview.describe(include="all").T)

        if st.button("تحميل الملف كاملاً إلى التطبيق"):
            try:
                uploaded_file.seek(0)
                df = read_full(uploaded_file, ftype, sheet_name=st.session_state.get("sheet"))
                st.session_state.df = df.copy()
                st.session_state.original_df = df.copy()
                st.success(f"تم تحميل البيانات بنجاح — الصفوف: {df.shape[0]}، الأعمدة: {df.shape[1]}")
            except Exception as e:
                st.error("فشل تحميل الملف: " + str(e))

# ---------------------------
# Page: Cleaning
# ---------------------------
if page == "تنظيف":
    st.header("تنظيف البيانات")
    if st.session_state.df is None:
        st.warning("لا يوجد بيانات محملة. اذهب إلى 'رفع البيانات' أولاً.")
    else:
        df = st.session_state.df

        st.subheader("ملخص سريع (مثل pandas.info بصيغة مبسطة)")
        info = df_info_summary(df)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("الصفوف", info["rows"])
            st.metric("الأعمدة", info["cols"])
            st.metric("الذاكرة (MB)", f"{info['memory_mb']:.2f}")
        with col2:
            df_cols = pd.DataFrame(info["columns"])
            st.dataframe(df_cols)

        st.markdown("---")
        st.subheader("التعامل مع القيم المفقودة")
        null_cols = [c for c in df.columns if df[c].isna().sum() > 0]
        if len(null_cols) == 0:
            st.success("لا يوجد قيم مفقودة")
        else:
            sel_cols = st.multiselect("اختر الأعمدة لتنظيف القيم المفقودة (اختياري)", null_cols, default=null_cols)
            strategy = st.radio("اختر الاستراتيجية", ("حذف الصفوف التي تحتوي مفاتيح مفقودة", "ملء بالمتوسط (numeric)", "ملء بالوسيط (numeric)", "ملء بالقيمة الشائعة (mode)"))
            if st.button("تطبيق تنظيف القيم المفقودة"):
                before = df.shape[0]
                if strategy == "حذف الصفوف التي تحتوي مفاتيح مفقودة":
                    df = df.dropna(subset=sel_cols)
                elif strategy == "ملء بالمتوسط (numeric)":
                    for c in sel_cols:
                        if pd.api.types.is_numeric_dtype(df[c]):
                            df[c] = df[c].fillna(df[c].mean())
                elif strategy == "ملء بالوسيط (numeric)":
                    for c in sel_cols:
                        if pd.api.types.is_numeric_dtype(df[c]):
                            df[c] = df[c].fillna(df[c].median())
                else:
                    for c in sel_cols:
                        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else "")
                st.session_state.df = df
                st.success(f"تم تطبيق الاستراتيجية — عدد الصفوف قبل: {before} بعد: {df.shape[0]}")

        st.markdown("---")
        st.subheader("تغيير أنواع الأعمدة سهلة")
        col_to_convert = st.selectbox("اختر عمودًا لتحويل نوعه (اختياري)", options=[None] + list(df.columns))
        if col_to_convert:
            target_type = st.selectbox("اختر النوع الهدف", ("numeric", "datetime", "category", "text"))
            if st.button("تطبيق التحويل"):
                try:
                    if target_type == "numeric":
                        df[col_to_convert] = pd.to_numeric(df[col_to_convert], errors="coerce")
                    elif target_type == "datetime":
                        df[col_to_convert] = pd.to_datetime(df[col_to_convert], errors="coerce")
                    elif target_type == "category":
                        df[col_to_convert] = df[col_to_convert].astype("category")
                    else:
                        df[col_to_convert] = df[col_to_convert].astype(str)
                    st.session_state.df = df
                    st.success("تم تحويل النوع بنجاح")
                except Exception as e:
                    st.error("خطأ أثناء التحويل: " + str(e))

        st.markdown("---")
        st.subheader("إزالة التكرارات")
        if st.button("إزالة التكرارات (drop_duplicates)"):
            before = df.shape[0]
            df = df.drop_duplicates()
            st.session_state.df = df
            st.success(f"تمت إزالة التكرارات — الصفوف قبل: {before}, بعد: {df.shape[0]}")

        st.markdown("---")
        st.subheader("إدارة القيم الشاذة (Outliers)")
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 0:
            st.info("لا توجد أعمدة رقمية لتحليل القيم الشاذة")
        else:
            chosen = st.multiselect("اختر أعمدة رقمية لفحص القيم الشاذة", numeric_cols, default=numeric_cols[:1])
            outlier_action = st.selectbox("ماذا تفعل بالقيم الشاذة؟", ("لم أفعل شيئًا (فقط تحديد)", "إزالة الصفوف الشاذة حسب IQR", "استبدالها بالحد الأدنى/الأقصى المقبول"))
            if st.button("تطبيق إدارة الشاذات"):
                mask_total = pd.Series(False, index=df.index)
                for c in chosen:
                    mask = simple_outlier_mask(df[c].dropna(), method="iqr", factor=1.5)
                    # align mask index with df
                    mask = df[c].index.isin(mask.index) & mask.reindex(df.index, fill_value=False)
                    mask_total = mask_total | mask
                if outlier_action == "لم أفعل شيئًا (فقط تحديد)":
                    st.warning(f"المجموع التقريبي للقيم الشاذة المكتشفة: {mask_total.sum()}")
                elif outlier_action == "إزالة الصفوف الشاذة حسب IQR":
                    before = df.shape[0]
                    df = df.loc[~mask_total]
                    st.session_state.df = df
                    st.success(f"تمت إزالة {before - df.shape[0]} صفوف شاذة")
                else:
                    # replace with clip boundaries
                    for c in chosen:
                        q1 = df[c].quantile(0.25)
                        q3 = df[c].quantile(0.75)
                        iqr = q3 - q1
                        low = q1 - 1.5 * iqr
                        high = q3 + 1.5 * iqr
                        df[c] = df[c].clip(lower=low, upper=high)
                    st.session_state.df = df
                    st.success("تم استبدال القيم الشاذة بالحدود المسموح بها (clip)")

        st.markdown("---")
        st.button("حفظ نسخة أصلية احتياطية", on_click=lambda: st.session_state.update({"original_df": st.session_state.df.copy()}))

# ---------------------------
# Page: Analysis
# ---------------------------
if page == "تحليل":
    st.header("التحليل السريع")
    if st.session_state.df is None:
        st.warning("لا يوجد بيانات. ارفع بياناتك أولًا.")
    else:
        df = st.session_state.df
        st.subheader("إحصائيات وصفية")
        st.dataframe(df.describe(include="all").T)

        st.markdown("---")
        st.subheader("ارتباط القيم (Correlation Heatmap)")
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation matrix")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("لا توجد أعمدة رقمية كافية لعرض الـ correlation.")

        st.markdown("---")
        st.subheader("أهم 5 رؤى تلقائية (Automatic Insights)")
        # Simple heuristics
        insights = []
        # high nulls
        for c in df.columns:
            null_pct = df[c].isna().mean()
            if null_pct > 0.3:
                insights.append(f"العمود '{c}' يحتوي على {null_pct:.0%} قيم مفقودة — فكر في تنظيفه أو حذفه.")
        # low variance
        for c in numeric_cols:
            if df[c].nunique() <= 2:
                insights.append(f"العمود '{c}' ذو تباين منخفض جدًا ({df[c].nunique()} قيم فريدة).")
        # correlations
        if len(numeric_cols) >= 2:
            corr_abs = corr.abs()
            high_pairs = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    a = numeric_cols[i]; b = numeric_cols[j]
                    if abs(corr.loc[a,b]) > 0.7:
                        high_pairs.append((a,b,corr.loc[a,b]))
            for a,b,v in high_pairs:
                insights.append(f"علاقة قوية بين '{a}' و '{b}' (corr={v:.2f}).")
        if len(insights) == 0:
            st.success("لا رؤى حرجة تم اكتشافها تلقائيًا. البيانات تبدو متوازنة.")
        else:
            for ins in insights[:8]:
                st.info(ins)

# ---------------------------
# Page: Models
# ---------------------------
if page == "نماذج":
    st.header("نماذج بسيطة (Classification / Regression)")
    if st.session_state.df is None:
        st.warning("لا يوجد بيانات. ارفع بياناتك أولًا.")
    else:
        df = st.session_state.df.copy()
        st.subheader("إعداد النموذج")
        target = st.selectbox("اختر العمود الهدف (target)", options=[None] + list(df.columns))
        if target:
            features = st.multiselect("اختر أعمدة الميزات (features) أو اترك تلقائيًا", options=[c for c in df.columns if c != target], default=[c for c in df.columns if c != target][:5])
            task = st.radio("نوع المهمة", ("تصنيف Classification", "تنبؤ Regression"))
            test_size = st.slider("نسبة بيانات الاختبار", 0.05, 0.5, 0.2)
            if st.button("تدريب النموذج"):
                # prepare data
                X = df[features].copy()
                y = df[target].copy()
                # simple encoding for categorical
                for col in X.select_dtypes(include=["object","category"]):
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                if y.dtype == "object" or y.dtype.name == "category" or y.dtype == "bool":
                    y_enc = LabelEncoder().fit_transform(y.astype(str))
                else:
                    y_enc = y.values
                # drop rows with nans
                mask = pd.concat([X, pd.Series(y_enc, index=X.index)], axis=1).dropna().index
                X = X.loc[mask]
                y_enc = pd.Series(y_enc, index=mask)
                X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=42)
                st.write("أحجام البيانات:", X_train.shape, X_test.shape)

                if task == "تصنيف Classification":
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    st.success(f"Accuracy: {acc:.3f}")
                    # confusion
                    cm = confusion_matrix(y_test, preds)
                    st.write("Confusion Matrix")
                    st.write(cm)
                    # feature importance
                    fi = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
                    st.write(fi.head(10))
                    # store model if needed
                    st.session_state["last_model"] = model
                else:
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    mse = mean_squared_error(y_test, preds)
                    st.success(f"MSE: {mse:.3f}")
                    fi = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
                    st.write(fi.head(10))
                    st.session_state["last_model"] = model

# ---------------------------
# Page: Plot
# ---------------------------
if page == "رسم":
    st.header("رسم البيانات بسهولة")
    if st.session_state.df is None:
        st.warning("لا يوجد بيانات لعرضها.")
    else:
        df = st.session_state.df
        st.subheader("اختيارات الرسم")
        cols = list(df.columns)
        x = st.selectbox("المحور X", options=[None] + cols)
        y = st.selectbox("المحور Y (اختياري)", options=[None] + cols)
        chart_type = st.selectbox("نوع الرسم", ("Scatter", "Line", "Bar", "Histogram", "Box"))
        if st.button("إنشاء الرسم"):
            try:
                if chart_type == "Scatter":
                    fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
                elif chart_type == "Line":
                    fig = px.line(df, x=x, y=y, title=f"{y} over {x}")
                elif chart_type == "Bar":
                    fig = px.bar(df, x=x, y=y, title=f"{y} by {x}")
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x, title=f"Distribution of {x}")
                else:
                    fig = px.box(df, x=x, y=y, title=f"Boxplot of {y} by {x}")
                st.plotly_chart(fig, use_container_width=True)
                # download image
                buf = fig.to_image(format="png")
                st.download_button("تحميل الرسم PNG", data=buf, file_name="chart.png", mime="image/png")
            except Exception as e:
                st.error("خطأ في إنشاء الرسم: " + str(e))

# ---------------------------
# Page: Export
# ---------------------------
if page == "تصدير":
    st.header("تصدير ومشاركة")
    if st.session_state.df is None:
        st.warning("لا توجد بيانات لتصديرها.")
    else:
        df = st.session_state.df
        st.write("معاينة أخيرة")
        st.dataframe(df.head(50))

        st.download_button("تحميل كـ CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="data_export.csv", mime="text/csv")
        st.download_button("تحميل كـ Excel", data=to_excel_bytes(df), file_name="data_export.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("تحميل كـ JSON", data=df.to_json(orient="records").encode("utf-8"), file_name="data_export.json", mime="application/json")

        st.markdown("---")
        st.info("إذا أردت مشاركة التطبيق: أنشره على Streamlit Cloud، وشارك الرابط.\nلأمان أعلى استخدم مصادقة (مثل OAuth) وHTTPS.")
