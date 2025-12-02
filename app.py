# app.py
"""
Data Analysis Hub - Full-featured Streamlit app
Features:
- Upload CSV / Excel
- Preview + keep original copy (df_original)
- Cleaning toolbox with explanations + auto-suggest cleaning
- EDA: descriptive, diagnostic, correlation, PCA, feature selection, hypothesis testing
- Train/Test split (preserve original)
- Visualization (Plotly + Seaborn) with quick interpretation hints
- Floating Code Editor (write & exec Python on df_working)
- AutoML (tries multiple models for classification/regression/regression)
- Undo/history & local checkpoints
- Export final code & trained model (pickle)
- Designed for good UX: step-by-step but fully flexible
"""

import streamlit as st
st.set_page_config(page_title="Data Analysis Hub", layout="wide", initial_sidebar_state="expanded")

# ---- Imports ----
import pandas as pd
import numpy as np
import io, os, json, pickle, textwrap
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from scipy.stats import ttest_ind
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

# ---- Constants ----
APP_STATE_DIR = "./.dah_state"
os.makedirs(APP_STATE_DIR, exist_ok=True)

REQUIRED_MODELS = {
    "classification": {
        "LogisticRegression": LogisticRegression,
        "RandomForest": RandomForestClassifier,
        "GradientBoosting": GradientBoostingClassifier,
        "DecisionTree": DecisionTreeClassifier,
        "KNeighbors": KNeighborsClassifier
    },
    "regression": {
        "LinearRegression": LinearRegression,
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "DecisionTree": DecisionTreeRegressor,
        "KNeighbors": KNeighborsRegressor
    },
    "clustering": {
        "KMeans": KMeans
    }
}

CLEANING_FUNCTIONS = {
    "dropna()": "حذف الصفوف أو الأعمدة التي تحتوي على قيم مفقودة (NaN).",
    "fillna()": "ملء القيم المفقودة بقيمة معينة مثل المتوسط أو القيمة الثابتة.",
    "drop_duplicates()": "حذف الصفوف المكررة من البيانات.",
    "astype()": "تحويل نوع البيانات لعمود معين (مثل نص → رقم).",
    "replace()": "استبدال قيم معينة بقيم أخرى.",
    "str.strip()": "حذف المسافات البيضاء من النص.",
    "str.lower()": "تحويل النصوص إلى حروف صغيرة.",
    "str.upper()": "تحويل النصوص إلى حروف كبيرة.",
    "apply()": "تطبيق دالة مخصصة على العمود لتنظيف أو تحويل القيم.",
    "rename()": "إعادة تسمية الأعمدة.",
    "split()/join()": "تقسيم نصوص ودمجها داخل الأعمدة.",
    "filter()": "تصفية الأعمدة أو الصفوف وفق شرط.",
    "isnull()/notnull()": "كشف القيم المفقودة أو غير المفقودة."
}

# ---- Session state init ----
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_working' not in st.session_state:
    st.session_state.df_working = None
if 'history' not in st.session_state:
    st.session_state.history = []  # list of dicts {time, action, df_snapshot_csv}
if 'checkpoints' not in st.session_state:
    st.session_state.checkpoints = []
if 'split' not in st.session_state:
    st.session_state.split = {}
if 'pipeline_log' not in st.session_state:
    st.session_state.pipeline_log = []  # user actions and code blocks
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = {}
if 'last_exec_output' not in st.session_state:
    st.session_state.last_exec_output = None

# ---- Helpers ----
def save_checkpoint(state, name_prefix="cp"):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{name_prefix}_{ts}.json"
    path = os.path.join(APP_STATE_DIR, filename)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str)
        st.session_state.checkpoints.insert(0, filename)
        return filename
    except Exception as e:
        st.warning(f"خطأ عند حفظ النسخة: {e}")
        return None

def snapshot_df(df):
    return df.to_csv(index=False)

def push_history(action):
    entry = {
        "time": str(datetime.utcnow()),
        "action": action,
        "df_csv": snapshot_df(st.session_state.df_working) if st.session_state.df_working is not None else None
    }
    st.session_state.history.append(entry)
    # keep limited history
    if len(st.session_state.history) > 50:
        st.session_state.history.pop(0)

def restore_from_history(index=-1):
    if not st.session_state.history:
        st.warning("لا توجد تاريخ للتراجع.")
        return
    entry = st.session_state.history[index]
    if entry.get("df_csv"):
        st.session_state.df_working = pd.read_csv(io.StringIO(entry["df_csv"]))
        st.success(f"تم استعادة الحالة: {entry['action']} عند {entry['time']}")
    else:
        st.warning("لا تحتوي هذه النقطة على بيانات لاستعادة.")

def safe_exec(user_code, globals_map):
    """
    Execute user code in provided globals_map context.
    Returns output or exception string.
    """
    try:
        # prepare local namespace
        loc = {}
        exec(user_code, globals_map, loc)
        # capture common variables like df_working, result, plt etc.
        output = {}
        # collect df_working if modified
        if "df_working" in globals_map:
            output["df_working"] = globals_map["df_working"]
        output["locals"] = {k: v for k, v in loc.items() if k not in ("__builtins__",)}
        st.session_state.last_exec_output = output
        return {"ok": True, "output": output}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ---- Layout ----
# Top header and main description
st.markdown("<h1 style='text-align:center; color:#4A90E2;'>حلّل بياناتك بسرعة — Data Analysis Hub</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>واجهة مرنة خطوة بخطوة، أو نفّذ أي كود في أي مرحلة — دون المساس بالبيانات الأصلية.</p>", unsafe_allow_html=True)
st.write("---")

# ---- Sidebar: Upload & Main Controls ----
with st.sidebar:
    st.header("1. البيانات & النسخ")
    uploaded_file = st.file_uploader("ارفع ملف CSV أو Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.df_original = df.copy()
            st.session_state.df_working = df.copy()
            push_history("Uploaded data")
            cp = {"meta": "upload", "time": str(datetime.utcnow())}
            cp["df_csv"] = snapshot_df(df)
            save_checkpoint(cp, name_prefix="upload")
            st.success("تم رفع الملف وحفظ نسخة أصلية.")
            st.write(f"شكل البيانات: {df.shape[0]} صف × {df.shape[1]} عمود")
        except Exception as e:
            st.error(f"خطأ في قراءة الملف: {e}")

    if st.button("🔁 تراجع (Undo آخر خطوة)"):
        if st.session_state.history:
            # pop last (which is current state) and restore previous
            if len(st.session_state.history) >= 2:
                st.session_state.history.pop()  # remove current
                restore_from_history(-1)
            else:
                st.warning("لا توجد خطوة أقدم للتراجع إليها.")
        else:
            st.warning("لا توجد تاريخ للتراجع.")

    st.markdown("---")
    st.header("2. نقاط الاستعادة")
    if st.session_state.checkpoints:
        sel_cp = st.selectbox("استعادة من نسخة محفوظة:", options=st.session_state.checkpoints)
        if st.button("استعادة النسخة"):
            try:
                with open(os.path.join(APP_STATE_DIR, sel_cp), "r", encoding="utf-8") as f:
                    cp = json.load(f)
                if cp.get("df_csv"):
                    st.session_state.df_working = pd.read_csv(io.StringIO(cp["df_csv"]))
                    push_history(f"restore:{sel_cp}")
                    st.success("تم استعادة النسخة المحفوظة.")
            except Exception as e:
                st.error(f"فشل استعادة النسخة: {e}")

    st.markdown("---")
    st.header("3. إجراءات سريعة")
    if st.button("حفظ نقطة استعادة الآن"):
        cp = {"meta": "manual_save", "time": str(datetime.utcnow())}
        cp["df_csv"] = snapshot_df(st.session_state.df_working) if st.session_state.df_working is not None else ""
        fn = save_checkpoint(cp, name_prefix="manual")
        if fn:
            st.success(f"تم حفظ نسخة: {fn}")

    if st.button("تفريغ كل الحالات (Reset app state)"):
        st.session_state.df_original = None
        st.session_state.df_working = None
        st.session_state.history = []
        st.session_state.checkpoints = []
        st.session_state.pipeline_log = []
        st.success("تم تفريغ الحالة.")

# ---- Main Tabs: EDA / Cleaning / Analysis / Split / Visualize / AutoML / Code Editor / Export
tabs = st.tabs(["استكشاف (Preview)", "تنظيف (Cleaning)", "تحليل (Analysis)", "تقسيم (Split)", "تصوير (Visualize)", "AutoML", "وحدة الأكواد (Code)", "تصدير / تنزيل"])

# ---------------- Tab 1: Preview ----------------
with tabs[0]:
    st.header("استكشاف عام — Preview")
    if st.session_state.df_working is None:
        st.info("ارفع ملفًا لبدء الاستكشاف.")
    else:
        df = st.session_state.df_working
        st.subheader("المعاينة (أول 10 صفوف)")
        st.dataframe(df.head(10))

        st.subheader("معلومات سريعة")
        c1, c2, c3 = st.columns(3)
        c1.metric("صفوف", df.shape[0])
        c2.metric("أعمدة", df.shape[1])
        c3.metric("القيم المفقودة الكلية", int(df.isnull().sum().sum()))

        if st.checkbox("عرض وصف كامل للبيانات (describe + dtypes)"):
            st.write(df.describe(include='all'))
            st.write(df.dtypes)

        if st.button("اقتراح تنظيف تلقائي"):
            # simple suggestions
            suggestions = []
            if df.isnull().sum().sum() > 0:
                suggestions.append("يوجد قيم مفقودة — اقترح fillna() أو dropna() حسب العمود.")
            if df.duplicated().sum() > 0:
                suggestions.append("يوجد صفوف مكررة — اقترح drop_duplicates().")
            text_cols = df.select_dtypes(include="object").columns.tolist()
            if text_cols:
                suggestions.append(f"أعمدة نصية قد تحتاج strip() أو lower(): {text_cols[:5]}")
            st.write("اقتراحات:")
            for s in suggestions:
                st.write("- ", s)

# ---------------- Tab 2: Cleaning ----------------
with tabs[1]:
    st.header("🧹 تنظيف البيانات")
    if st.session_state.df_working is None:
        st.info("يرجى رفع ملف أولًا.")
    else:
        df = st.session_state.df_working

        st.write("مرّر على أي دالة لقراءة شرحها.")
        func_choice = st.selectbox("اختر دالة تنظيف:", options=list(CLEANING_FUNCTIONS.keys()), format_func=lambda x: x)
        st.caption(CLEANING_FUNCTIONS[func_choice])

        st.markdown("**خيارات سريعة للتنظيف:**")
        col1, col2, col3 = st.columns(3)
        if col1.button("ملء القيم المفقودة بالوسيط (median)"):
            push_history("fillna_median")
            num_cols = st.session_state.df_working.select_dtypes(include=np.number).columns
            st.session_state.df_working[num_cols] = st.session_state.df_working[num_cols].fillna(st.session_state.df_working[num_cols].median())
            st.success("تم ملء القيم المفقودة للأعمدة العددية بالوسيط.")
        if col2.button("حذف الصفوف التي تحتوي NA"):
            push_history("dropna_rows")
            st.session_state.df_working = st.session_state.df_working.dropna(axis=0)
            st.success("تم حذف الصفوف التي تحتوي NA.")
        if col3.button("حذف الصفوف المكررة"):
            push_history("drop_duplicates")
            st.session_state.df_working = st.session_state.df_working.drop_duplicates()
            st.success("تم حذف الصفوف المكررة.")

        st.markdown("**تنظيف نصوص سريع**")
        text_cols = st.session_state.df_working.select_dtypes(include="object").columns.tolist()
        sel_text = st.multiselect("اختر أعمدة نصية لتطبيق strip() وlower():", options=text_cols)
        if st.button("تطبيق تنظيف النصوص"):
            if sel_text:
                push_history("clean_text")
                for c in sel_text:
                    st.session_state.df_working[c] = st.session_state.df_working[c].astype(str).str.strip().str.lower()
                st.success(f"تم تنظيف الأعمدة: {sel_text}")
            else:
                st.info("لم تختَر أعمدة نصية.")

        st.markdown("---")
        st.subheader("تخصيص عمليّة تنظيف (Advanced)")
        st.write("يمكنك كتابة دالة تنظيف مخصصة في محرّر الأكواد وتشغيلها على df_working.")

# ---------------- Tab 3: Analysis ----------------
with tabs[2]:
    st.header("📊 تحليل البيانات")
    if st.session_state.df_working is None:
        st.info("ارفع بيانات لتبدأ التحليل.")
    else:
        df = st.session_state.df_working
        st.write("اختر نوع التحليل ثم الأعمدة المستهدفة.")
        analysis_type = st.selectbox("اختر نوع التحليل:", [
            "التحليل الوصفي - Descriptive",
            "التحليل التشخيصي - Diagnostic",
            "تحليل الارتباطات - Correlation",
            "تحليل العوامل (PCA) - Factor",
            "اختيار الميزات - Feature Selection",
            "اختبار الفرضيات - Hypothesis Testing"
        ])
        cols = st.multiselect("اختر أعمدة للتحليل:", options=df.columns.tolist())
        if not cols:
            st.info("اختر عمودًا واحدًا على الأقل للمضي قدمًا.")
        else:
            if "Descriptive" in analysis_type:
                st.subheader("التحليل الوصفي")
                st.dataframe(df[cols].describe(include='all'))
            if "Diagnostic" in analysis_type:
                st.subheader("التحليل التشخيصي")
                st.write("Missing per column:")
                st.write(df[cols].isnull().sum())
                st.write("Basic distributions:")
                st.write(df[cols].describe(include='all'))
            if "Correlation" in analysis_type:
                st.subheader("Correlation Matrix")
                num_df = df[cols].select_dtypes(include=np.number)
                corr = num_df.corr()
                st.dataframe(corr)
                fig = px.imshow(corr, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
                # quick interpretation
                st.write("تفسير سريع:")
                high_corr = []
                for i in corr.columns:
                    for j in corr.columns:
                        if i!=j and abs(corr.loc[i,j])>0.7:
                            high_corr.append((i,j,corr.loc[i,j]))
                if high_corr:
                    st.success(f"وجدت علاقات عالية بين: {high_corr[:5]}")
                else:
                    st.info("لا توجد علاقات قوية (>0.7) بين الأعمدة المختارة.")
            if "PCA" in analysis_type:
                st.subheader("PCA - تحليل العوامل")
                num_df = df[cols].select_dtypes(include=np.number).dropna()
                if num_df.shape[1] < 2:
                    st.error("أحتاج على الأقل عمودين رقميين لـ PCA.")
                else:
                    pca = PCA(n_components=min(3, num_df.shape[1]))
                    comps = pca.fit_transform(num_df)
                    comp_df = pd.DataFrame(comps, columns=[f"PC{i+1}" for i in range(comps.shape[1])])
                    st.write("Explained variance ratio:", pca.explained_variance_ratio_)
                    fig = px.scatter(comp_df, x="PC1", y="PC2")
                    st.plotly_chart(fig, use_container_width=True)
            if "Feature Selection" in analysis_type:
                st.subheader("اختيار الميزات")
                target_col = st.selectbox("اختر العمود الهدف (Target):", options=df.columns.tolist(), index=0)
                k = st.slider("كم عدد الميزات تريد اختيارها؟", 1, max(1, len(df.columns)-1), 3)
                if st.button("تشغيل اختيار الميزات"):
                    try:
                        X = df.drop(columns=[target_col]).select_dtypes(include=np.number).dropna()
                        y = df[target_col]
                        selector = SelectKBest(score_func=(f_classif if y.dtype.kind in 'biufc' else f_classif), k=k)
                        selector.fit(X, y)
                        scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
                        st.write(scores.head(k))
                    except Exception as e:
                        st.error(f"فشل اختيار الميزات: {e}")
            if "Hypothesis Testing" in analysis_type:
                st.subheader("اختبار الفرضيات (T-test)")
                c1 = st.selectbox("اختر العمود الأول:", options=df.columns.tolist(), index=0)
                c2 = st.selectbox("اختر العمود الثاني:", options=df.columns.tolist(), index=0)
                if st.button("تشغيل T-test"):
                    try:
                        stat, p = ttest_ind(df[c1].dropna(), df[c2].dropna())
                        st.write(f"T-stat: {stat:.4f}  P-value: {p:.6f}")
                        if p < 0.05:
                            st.success("يوجد فرق ذو دلالة إحصائية (p < 0.05).")
                        else:
                            st.info("لا يوجد فرق ذو دلالة إحصائية.")
                    except Exception as e:
                        st.error(f"فشل الاختبار: {e}")

# ---------------- Tab 4: Split ----------------
with tabs[3]:
    st.header("🔀 تقسيم البيانات (Train / Test)")
    if st.session_state.df_working is None:
        st.info("أرفع البيانات أولا.")
    else:
        df = st.session_state.df_working
        st.write("اختر العمود الهدف (Target) والأعمدة المميزة (Features).")
        target = st.selectbox("اختر الهدف:", options=df.columns.tolist())
        features = st.multiselect("اختر المزايا (اترك فارغا لاختيار كل الأعمدة العددية):", options=[c for c in df.columns.tolist() if c!=target])
        test_size = st.slider("نسبة الاختبار (test_size):", 0.05, 0.5, 0.2)
        stratify_option = None
        if st.button("نفّذ التقسيم"):
            # prepare X,y
            if not features:
                X = df.drop(columns=[target]).select_dtypes(include=np.number)
            else:
                X = df[features]
            y = df[target]
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=(y if y.nunique()<50 else None))
            except Exception:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            st.session_state.split = {
                "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
                "features": X.columns.tolist(), "target": target
            }
            push_history("split_data")
            st.success(f"تم التقسيم: {len(X_train)} تدريب / {len(X_test)} اختبار")
            st.write("مميزات:", st.session_state.split["features"])

# ---------------- Tab 5: Visualize ----------------
with tabs[4]:
    st.header("📈 تصوير البيانات")
    if st.session_state.df_working is None:
        st.info("أرفع بيانات لتصويرها.")
    else:
        df = st.session_state.df_working
        viz_type = st.selectbox("اختر نوع الرسم:", ["Histogram", "Scatter", "Line", "Bar", "Box", "Pairplot", "Correlation Heatmap"])
        if viz_type in ["Histogram", "Box", "Bar", "Line", "Scatter"]:
            col_x = st.selectbox("المحور X / العمود:", options=df.columns.tolist())
            if viz_type == "Scatter":
                col_y = st.selectbox("المحور Y:", options=[c for c in df.columns if c!=col_x])
            else:
                col_y = None
            if st.button("عرض الرسم"):
                try:
                    if viz_type == "Histogram":
                        fig = px.histogram(df, x=col_x, marginal="box")
                    elif viz_type == "Box":
                        fig = px.box(df, y=col_x)
                    elif viz_type == "Bar":
                        ct = df[col_x].value_counts().reset_index()
                        ct.columns = [col_x, "count"]
                        fig = px.bar(ct, x=col_x, y="count")
                    elif viz_type == "Line":
                        fig = px.line(df, x=df.index, y=col_x)
                    elif viz_type == "Scatter":
                        fig = px.scatter(df, x=col_x, y=col_y)
                    st.plotly_chart(fig, use_container_width=True)
                    # quick interpretation
                    st.write("تفسير مبسّط:")
                    if viz_type == "Histogram":
                        st.write("تحقق من الانحراف أو القيم الشاذة (outliers) في التوزيع.")
                    if viz_type == "Scatter":
                        st.write("انظر إلى العلاقة العامة بين المتغيرين إن وُجدت (خطية/غير خطية).")
                except Exception as e:
                    st.error(f"خطأ في الرسم: {e}")
        elif viz_type == "Pairplot":
            sel = st.multiselect("اختر أعمدة:", options=df.select_dtypes(include=np.number).columns.tolist(), default=df.select_dtypes(include=np.number).columns.tolist()[:4])
            if st.button("عرض Pairplot"):
                try:
                    fig = sns.pairplot(df[sel].dropna().sample(min(500, len(df))))
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"فشل: {e}")
        else:  # Correlation Heatmap
            num = df.select_dtypes(include=np.number)
            if num.shape[1] < 2:
                st.info("تحتاج على الأقل عمودين عدديين للخرائط الارتباطية.")
            else:
                corr = num.corr()
                fig = px.imshow(corr, text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
                st.write("تفسير: قيم قريبة من 1 أو -1 تدل على ارتباط قوي.")

# ---------------- Tab 6: AutoML ----------------
with tabs[5]:
    st.header("🤖 AutoML — تجربة نماذج متعددة وقياس الأداء")
    if st.session_state.df_working is None:
        st.info("أرفع بيانات وغيّرها ثم عد هنا.")
    else:
        df = st.session_state.df_working
        task = st.selectbox("نوع المهمة:", ["Classification", "Regression", "Clustering"])
        if task in ["Classification", "Regression"]:
            # choose target
            target = st.selectbox("اختر العمود الهدف:", options=df.columns.tolist())
            features = st.multiselect("اختر الأعمدة Features (اترك فارغًا للأعمدة العددية التلقائية):", options=[c for c in df.columns if c!=target])
            test_size = st.slider("نسبة الاختبار:", 0.05, 0.4, 0.2)
            if st.button("تشغيل AutoML"):
                # prepare X,y
                if not features:
                    X = df.drop(columns=[target]).select_dtypes(include=np.number)
                else:
                    X = df[features]
                y = df[target]
                # Train/test split
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=(y if task=="Classification" and y.nunique()<50 else None))
                except Exception:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                # simple preprocessor: impute + scale for numeric, encode cats if any
                numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
                cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
                num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
                cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))]) if cat_cols else None
                preprocessor = ColumnTransformer(
                    transformers=[("num", num_pipe, numeric_cols)] + ([("cat", cat_pipe, cat_cols)] if cat_cols else []),
                    remainder="drop"
                )
                models_to_try = REQUIRED_MODELS["classification" if task=="Classification" else "regression"]
                results = []
                for name, cls in models_to_try.items():
                    try:
                        pipe = Pipeline([("pre", preprocessor), ("model", cls())])
                        pipe.fit(X_train, y_train)
                        preds = pipe.predict(X_test)
                        if task == "Classification":
                            acc = accuracy_score(y_test, preds)
                            f1 = f1_score(y_test, preds, average="weighted") if len(np.unique(y_test))>1 else f1_score(y_test, preds, average="macro")
                            results.append({"model": name, "acc": acc, "f1": f1, "estimator": pipe})
                        else:
                            mse = mean_squared_error(y_test, preds)
                            r2 = r2_score(y_test, preds)
                            results.append({"model": name, "mse": mse, "r2": r2, "estimator": pipe})
                    except Exception as e:
                        st.write(f"فشل نموذج {name}: {e}")
                # show results
                if results:
                    st.write("نتائج النماذج:")
                    if task == "Classification":
                        res_df = pd.DataFrame(results)[["model","acc","f1"]].sort_values("acc", ascending=False)
                        st.dataframe(res_df)
                        best = max(results, key=lambda r: r["acc"])
                    else:
                        res_df = pd.DataFrame(results)[["model","r2","mse"]].sort_values("r2", ascending=False)
                        st.dataframe(res_df)
                        best = max(results, key=lambda r: r["r2"])
                    st.success(f"أفضل نموذج: {best['model']}")
                    # save best
                    st.session_state.models_trained["best"] = best
                    # option to download model
                    if st.button("حفظ النموذج الأفضل (pickle)"):
                        fn = f"best_model_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.pkl"
                        with open(fn, "wb") as f:
                            pickle.dump(best["estimator"], f)
                        st.success(f"تم حفظ النموذج في {fn}")
                        with open(fn, "rb") as f:
                            st.download_button("⬇️ تنزيل النموذج (pickle)", data=f, file_name=fn)
                else:
                    st.info("لم يتم الحصول على نتائج صالحة.")

        else:  # Clustering
            n_clusters = st.slider("عدد المجموعات (n_clusters):", 2, 20, 5)
            numeric = df.select_dtypes(include=np.number).dropna()
            if numeric.shape[1] < 1:
                st.info("تحتاج أعمدة رقمية للـ Clustering.")
            else:
                if st.button("تشغيل KMeans"):
                    k = KMeans(n_clusters=n_clusters, random_state=42)
                    labels = k.fit_predict(numeric)
                    st.session_state.df_working["_cluster"] = labels
                    st.success("تم تنفيذ KMeans وإضافة عمود _cluster")
                    st.write(pd.Series(labels).value_counts())

# ---------------- Tab 7: Code Editor (Floating-like) ----------------
with tabs[6]:
    st.header("🧾 محرر الأكواد — نفّذ أي كود في أي مكان")
    st.write("يمكنك كتابة كود Python هنا وتشغيله على `df_working` أو أي جزء من الـ pipeline.")
    st.info("ملاحظة: التنفيذ سيؤثر على نسخة العمل df_working فقط (ما لم تُعدّل df_original يدويًا).")

    # Execution target
    exec_target = st.selectbox("نفّذ الكود على:", options=[
        "df_working (النسخة العاملة)",
        "قبل التحليل (pre)",
        "بعد التنظيف (post_clean)",
        "بعد التقسيم (post_split)",
        "النماذج (models)",
        "تحميل/تصدير (export)"
    ])

    code_area = st.text_area("اكتب كود Python هنا:", value=textwrap.dedent("""# مثال:
# df_working['new_col'] = df_working['some_numeric_col'] * 2
# def mark_outliers(df):
#     df['is_outlier'] = (df['new_col'] > df['new_col'].quantile(0.99))
#     return df
# df_working = mark_outliers(df_working)
pass
"""), height=250)

    run_col, save_col = st.columns([1,1])
    if run_col.button("تشغيل الكود الآن"):
        # prepare safe globals
        globals_map = {
            "pd": pd, "np": np, "plt": plt, "sns": sns, "px": px,
            "df_original": st.session_state.df_original,
            "df_working": st.session_state.df_working,
            "X_train": st.session_state.split.get("X_train"),
            "X_test": st.session_state.split.get("X_test"),
            "y_train": st.session_state.split.get("y_train"),
            "y_test": st.session_state.split.get("y_test"),
            # sklearn available
            "RandomForestClassifier": RandomForestClassifier,
            "RandomForestRegressor": RandomForestRegressor,
            "LinearRegression": LinearRegression,
            "LogisticRegression": LogisticRegression
        }
        res = safe_exec(code_area, globals_map)
        if res["ok"]:
            # update df_working if changed
            if "df_working" in globals_map and globals_map["df_working"] is not None:
                st.session_state.df_working = globals_map["df_working"]
                push_history("exec_code")
                st.success("تم تنفيذ الكود وتحديث df_working.")
            st.write("ناتج التنفيذ (لو موجود):")
            st.write(res["output"].get("locals", {}))
        else:
            st.error(f"خطأ في التنفيذ: {res['error']}")

    if save_col.button("حفظ كتلة كجزء من البايبلاين"):
        block = {
            "time": str(datetime.utcnow()),
            "target": exec_target,
            "code": code_area
        }
        st.session_state.pipeline_log.append(block)
        push_history("save_code_block")
        st.success("تم حفظ كتلة الكود في سجل البايبلاين.")

    if st.session_state.pipeline_log:
        st.markdown("**سجل الأكواد المُحفوظة:**")
        for i, b in enumerate(st.session_state.pipeline_log[::-1]):
            st.markdown(f"- [{b['time']}] target={b['target']} — code preview: `{b['code'][:80].replace('\\n',' ')}...`")
            if st.button(f"تشغيل هذه الكتلة #{len(st.session_state.pipeline_log)-i-1}"):
                # run the block
                globals_map = {
                    "pd": pd, "np": np, "plt": plt, "sns": sns, "px": px,
                    "df_original": st.session_state.df_original,
                    "df_working": st.session_state.df_working,
                    "X_train": st.session_state.split.get("X_train"),
                    "X_test": st.session_state.split.get("X_test"),
                    "y_train": st.session_state.split.get("y_train"),
                    "y_test": st.session_state.split.get("y_test"),
                }
                res = safe_exec(b["code"], globals_map)
                if res["ok"]:
                    if "df_working" in globals_map and globals_map["df_working"] is not None:
                        st.session_state.df_working = globals_map["df_working"]
                        push_history("exec_saved_block")
                        st.success("تم تشغيل الكتلة وتحديث df_working.")
                else:
                    st.error(f"فشل تشغيل الكتلة: {res['error']}")

# ---------------- Tab 8: Export / Download ----------------
with tabs[7]:
    st.header("📦 تصدير و تنزيل")
    st.write("يمكنك تنزيل الكود الذي يكرر ما قمت به، أو تنزيل النموذج المدرب أو بيانات العمل.")
    if st.session_state.df_working is None:
        st.info("لا توجد بيانات للعملية بعد.")
    else:
        # Export working data
        buf = io.StringIO()
        st.session_state.df_working.to_csv(buf, index=False)
        st.download_button("⬇️ تنزيل نسخة البيانات المعدلة (CSV)", data=buf.getvalue(), file_name="df_working.csv", mime="text/csv")

        # Export pipeline log as script (generate python script)
        if st.button("إنشاء كود Python من السجل (تكرار الخطوات)"):
            # Build simple script
            script_lines = [
                "# Generated pipeline script from Data Analysis Hub",
                "import pandas as pd, numpy as np",
                "from sklearn.model_selection import train_test_split",
                ""
            ]
            script_lines.append("# Load data (user should replace path)")
            script_lines.append("df = pd.read_csv('your_data.csv')\n")
            for step in st.session_state.pipeline_log:
                script_lines.append("# --- Block saved at: " + step["time"])
                script_lines.append(step["code"])
                script_lines.append("\n")
            script_text = "\n".join(script_lines)
            st.download_button("⬇️ تنزيل كود pipeline (script.py)", data=script_text, file_name="pipeline_script.py", mime="text/x-python")
            st.code(script_text[:1000] + "\n\n# ... (full script available for download)")

        # Export best model if exists
        if st.session_state.models_trained.get("best"):
            best = st.session_state.models_trained["best"]
            if st.button("⬇️ تنزيل أفضل نموذج تم تدريبه (pickle)"):
                fn = f"best_model_{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.pkl"
                with open(fn, "wb") as f:
                    pickle.dump(best["estimator"], f)
                with open(fn, "rb") as f:
                    st.download_button("تحميل النموذج (pickle)", data=f, file_name=fn)

        # Export full app code (this file)
        if st.button("⬇️ تنزيل كود التطبيق الكامل (app.py)"):
            try:
                with open(__file__, "r", encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                # fallback: export a helpful message
                content = "# ضع هنا كود التطبيق أو افتح app.py محليًا لتحميل الكود."
            st.download_button("تحميل app.py", data=content, file_name="app.py", mime="text/x-python")

st.markdown("---")
st.caption("Data Analysis Hub — تم تجهيزه ليكون سهل الاستخدام، مرن، وقابل للتوسعة. إذا رغبت في إضافة AutoML أعمق (Bayesian tuning, ensembling متقدم, AutoCV) أو واجهة White-label وسجل مستخدمين، أخبرني وسأقوم بكتابته لك كخطوة ثانية.")
