import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import base64
from PIL import Image
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings('ignore')

# Data processing libraries
from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pycaret
from pycaret.regression import *
from pycaret.classification import *

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Page config
st.set_page_config(
    page_title="DataWizard Pro",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state initialization
if 'df' not in st.session_state:
    st.session_state.df = None
if 'clean_df' not in st.session_state:
    st.session_state.clean_df = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

class DataWizardPro:
    def __init__(self):
        self.progress = st.progress(0)
    
    def load_data(self):
        """Professional data loading with multiple formats"""
        st.markdown('<h1 class="main-header">🧙‍♂️ DataWizard Pro</h1>', unsafe_allow_html=True)
        st.markdown("**لوحة تحليل بيانات احترافية كاملة - تنظيف | استكشاف | تحليل | ML | تصور**")
        
        col1, col2 = st.columns([3,1])
        with col1:
            uploaded_file = st.file_uploader(
                "📁 رفع ملف البيانات (CSV, Excel, JSON)",
                type=['csv', 'xlsx', 'xls', 'json'],
                help="يدعم جميع الصيغ الشائعة"
            )
        with col2:
            st.info("**الميزات المتقدمة:**
✅ تنظيف تلقائي
✅ AutoML
✅ 25+ تصور
✅ حفظ النماذج")
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    df = pd.read_json(uploaded_file)
                
                st.session_state.df = df
                st.session_state.progress.progress(20)
                st.success(f"✅ تم رفع البيانات بنجاح!
📊 الأبعاد: {df.shape[0]:,} صف × {df.shape[1]} عمود")
                st.dataframe(df.head(), use_container_width=True)
                return True
            except Exception as e:
                st.error(f"خطأ في رفع الملف: {str(e)}")
        return False
    
    def auto_clean(self, df):
        """Advanced automated data cleaning"""
        st.markdown("### 🧹 التنظيف التلقائي المتقدم")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            drop_na = st.checkbox("🗑️ حذف القيم المفقودة", value=True)
        with col2:
            fill_strategy = st.selectbox("ملء القيم المفقودة", 
                                       ["متوسط", "وسيط", "أكثر تكرار", "ثابت"])
        with col3:
            drop_duplicates = st.checkbox("📋 حذف التكرارات", value=True)
        with col4:
            outlier_method = st.selectbox("المتطرفات", ["لا", "IQR", "Z-score"])
        
        df_clean = df.copy()
        
        # Missing values handling
        if drop_na:
            initial_na = df_clean.isnull().sum().sum()
            df_clean = df_clean.dropna()
            st.success(f"حُذفت {initial_na - df_clean.isnull().sum().sum():,} قيمة مفقودة")
        
        # Fill missing values
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        if fill_strategy != "ثابت" and len(numeric_cols) > 0:
            for col in numeric_cols:
                if df_clean[col].isnull().sum() > 0:
                    if fill_strategy == "متوسط":
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif fill_strategy == "وسيط":
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    else:
                        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        # Duplicates
        if drop_duplicates:
            initial_rows = len(df_clean)
            df_clean = df_clean.drop_duplicates()
            st.success(f"حُذفت {initial_rows - len(df_clean):,} صفوف مكررة")
        
        # Outliers (simplified IQR)
        if outlier_method == "IQR":
            for col in numeric_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        
        st.session_state.clean_df = df_clean
        st.session_state.progress.progress(50)
        return df_clean
    
    def eda_pro(self, df):
        """Professional EDA with multiple visualizations"""
        st.markdown("### 📊 الاستكشاف الاحترافي للبيانات (EDA)")
        
        col1, col2 = st.columns(2)
        with col1:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                col_selected = st.selectbox("اختر عمود رقمي", numeric_cols)
                fig_hist = px.histogram(df, x=col_selected, 
                                      title=f"توزيع {col_selected}",
                                      marginal="box")
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            if len(numeric_cols) >= 2:
                col1_sel = st.selectbox("المحور X", numeric_cols, index=0)
                col2_sel = st.selectbox("المحور Y", numeric_cols, index=1)
                fig_scatter = px.scatter(df, x=col1_sel, y=col2_sel,
                                       title=f"الارتباط بين {col1_sel} و {col2_sel}")
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Correlation heatmap
        if len(numeric_cols) >= 2:
            st.subheader("🔥 مصفوفة الارتباطات")
            corr = df[numeric_cols].corr()
            fig_heatmap = px.imshow(corr, aspect="auto", color_continuous_scale="RdBu_r")
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Auto EDA Report
        if st.button("⚡ تقرير EDA تلقائي شامل", type="primary"):
            with st.spinner("جاري إنشاء التقرير... قد يستغرق دقيقة"):
                profile = ProfileReport(df, title="تقرير DataWizard Pro", 
                                      explorative=True, minimal=False)
                st_profile_report(profile)
    
    def ml_pro(self, df):
        """Professional AutoML with PyCaret"""
        st.markdown("### 🤖 AutoML احترافي مع PyCaret")
        
        target_col = st.selectbox("🎯 العمود المستهدف", df.columns)
        task_type = st.radio("نوع المهمة", ["تصنيف", "تنبؤ"])
        
        if st.button("🚀 بدء AutoML!", type="primary"):
            with st.spinner("جاري تدريب 15+ نموذج تلقائياً..."):
                temp_df = df.copy()
                
                # Prepare data
                X = temp_df.drop(columns=[target_col])
                y = temp_df[target_col]
                
                # Handle categorical variables
                categorical_cols = X.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                
                # PyCaret setup
                if task_type == "تصنيف":
                    setup_df = pd.concat([X, y], axis=1)
                    setup_df.columns = [f"feature_{i}" if i != target_col else "target" 
                                      for i in range(len(setup_df.columns))]
                    s = setup(setup_df, target='target', session_id=123, silent=True)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    s = setup(X_train, target=y_train, session_id=123, silent=True)
                
                # Compare models
                best_model = compare_models(n_select=5)
                tuned_model = tune_model(best_model)
                final_model = finalize_model(tuned_model)
                
                st.session_state.models['best'] = final_model
                st.success("✅ تم تدريب أفضل النماذج تلقائياً!")
                
                # Show results
                st.subheader("🏆 نتائج أفضل 5 نماذج")
                models = pull()
                st.dataframe(models.head())
    
    def download_utils(self, df, filename="datawizard_results"):
        """Professional download utilities"""
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" class="success-box">📥 تحميل البيانات النظيفة</a>'
        st.markdown(href, unsafe_allow_html=True)

# Main App
@st.cache_data
def main():
    app = DataWizardPro()
    
    # Step 1: Load Data
    if app.load_data():
        df_raw = st.session_state.df
        
        # Step 2: Auto Clean
        st.session_state.progress.progress(30)
        clean_df = app.auto_clean(df_raw)
        
        # Step 3: Tabs for Professional Analysis
        tab1, tab2, tab3, tab4 = st.tabs(["📊 استكشاف", "🔍 تحليل متقدم", "🤖 AutoML", "📈 لوحة التحكم"])
        
        with tab1:
            app.eda_pro(clean_df)
        
        with tab2:
            app.ml_pro(clean_df)
        
        with tab3:
            st.subheader("🎛️ لوحة التحكم التنفيذية")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("عدد الصفوف", f"{len(clean_df):,}")
            with col2:
                st.metric("عدد الأعمدة", f"{len(clean_df.columns)}")
            with col3:
                missing_pct = clean_df.isnull().sum().sum() / (len(clean_df) * len(clean_df.columns)) * 100
                st.metric("نسبة القيم المفقودة", f"{missing_pct:.1f}%")
            with col4:
                st.metric("الذاكرة المستخدمة", f"{clean_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        with tab4:
            app.download_utils(clean_df)
        
        st.balloons()
        st.session_state.progress.progress(100)
    else:
        st.info("📤 يرجى رفع ملف البيانات لبدء التحليل")

if __name__ == "__main__":
    main()
