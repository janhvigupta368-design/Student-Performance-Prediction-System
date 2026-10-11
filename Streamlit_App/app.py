import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
import io

# ---------------------------------------------------------
# Page Configuration & UI Theme (Academic Modern Blue/White)
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduPredict | Student Performance Analytics & Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top Banner Gradient */
    .header-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 50%, #3b82f6 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.15);
    }
    .header-box h1 {
        color: white !important;
        margin: 0;
        font-weight: 700;
        font-size: 28px;
    }
    .header-box p {
        color: #e0e7ff;
        margin-top: 6px;
        font-size: 15px;
    }
    
    /* Active Dataset Badge */
    .dataset-badge {
        display: inline-block;
        padding: 6px 14px;
        background-color: #eff6ff;
        color: #1d4ed8;
        border: 1px solid #bfdbfe;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
        margin-bottom: 16px;
    }
    
    /* Metric Cards */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
        border-left: 4px solid #2563eb;
    }
    .stat-card-title {
        color: #64748b;
        font-size: 13px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-card-value {
        color: #0f172a;
        font-size: 24px;
        font-weight: 700;
        margin-top: 4px;
    }
    
    /* Info Card */
    .info-box {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        border: none;
        padding: 8px 18px;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #1d4ed8;
        color: white;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    
    /* Sidebar */
    .sidebar-brand {
        padding: 12px 0 20px 0;
        text-align: center;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 20px;
    }
    .sidebar-brand h2 {
        color: #1e3a8a;
        margin: 0;
        font-size: 20px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions & Resource Loaders
# ---------------------------------------------------------
@st.cache_data
def load_default_data():
    """Load the default UCI student dataset."""
    possible_paths = [
        os.path.join("Dataset", "student-mat.csv"),
        os.path.join("..", "Dataset", "student-mat.csv"),
        "student-mat.csv"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return pd.read_csv(path, sep=';')
            except Exception:
                return pd.read_csv(path)
    return None

@st.cache_resource
def load_trained_model():
    """Load the saved trained ML pipeline model."""
    possible_paths = [
        os.path.join("Model", "best_model.pkl"),
        os.path.join("..", "Model", "best_model.pkl"),
        "best_model.pkl"
    ]
    for path in possible_paths:
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                return None
    return None

def get_target_column(df):
    """Detect target grade column (e.g., G3, g3, FinalGrade) case-insensitively."""
    if df is None:
        return None
    for col in df.columns:
        if str(col).strip().upper() == 'G3':
            return col
    for col in df.columns:
        if str(col).strip().lower() in ['finalgrade', 'final_grade', 'grade', 'target']:
            return col
    return None

def detect_outliers_iqr(series):
    """Detect outliers using IQR rule."""
    numeric_s = pd.to_numeric(series, errors='coerce').dropna()
    if len(numeric_s) < 4:
        return 0
    q25, q75 = numeric_s.quantile(0.25), numeric_s.quantile(0.75)
    iqr = q75 - q25
    lower_bound = q25 - (1.5 * iqr)
    upper_bound = q75 + (1.5 * iqr)
    outliers = numeric_s[(numeric_s < lower_bound) | (numeric_s > upper_bound)]
    return len(outliers)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
default_df = load_default_data()
model = load_trained_model()

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

# Active Dataset Selection
if st.session_state.uploaded_data is not None:
    current_df = st.session_state.uploaded_data
    current_dataset_label = f"Uploaded Dataset ({st.session_state.uploaded_filename})"
    is_uploaded = True
else:
    current_df = default_df
    current_dataset_label = "Original UCI Dataset (student-mat.csv)"
    is_uploaded = False

# ---------------------------------------------------------
# Sidebar Navigation & Active Status
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <h2>🎓 EduPredict AI</h2>
            <p style="color:#64748b; font-size:12px; margin:0;">Academic Performance System</p>
        </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation Menu",
        [
            "Home Dashboard",
            "Upload Student Data",
            "Student Data Analysis",
            "Predict Student Performance",
            "Batch Prediction",
            "Model Performance"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📌 Active Dataset Status")
    if is_uploaded:
        st.markdown(f"**Source:** 📂 Uploaded")
        st.caption(f"File: `{st.session_state.uploaded_filename}`")
        if st.button("Reset to Default Dataset", use_container_width=True):
            st.session_state.uploaded_data = None
            st.session_state.uploaded_filename = None
            st.rerun()
    else:
        st.markdown(f"**Source:** 🏛️ Original Dataset")
        st.caption(f"student-mat.csv ({current_df.shape[0] if current_df is not None else 649} records)")
        
    st.markdown("---")
    st.caption("🤖 Model: Scikit-Learn Pipeline (Gradient Boosting / Random Forest)")


# =========================================================
# 1. HOME DASHBOARD
# =========================================================
if menu == "Home Dashboard":
    st.markdown("""
        <div class="header-box">
            <h1>🎓 Student Performance Analysis & Prediction System</h1>
            <p>Empower educators with data-driven academic analytics, dynamic visual exploration, and machine learning grade predictions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Dataset indicator
    st.markdown(f'<div class="dataset-badge">Active Dataset: {current_dataset_label}</div>', unsafe_allow_html=True)
    
    if current_df is not None:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-card-title">Total Records</div>
                    <div class="stat-card-value">{current_df.shape[0]:,}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-card-title">Total Features</div>
                    <div class="stat-card-value">{current_df.shape[1]}</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            missing_total = current_df.isnull().sum().sum()
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-card-title">Missing Values</div>
                    <div class="stat-card-value">{missing_total}</div>
                </div>
            """, unsafe_allow_html=True)
        home_target = get_target_column(current_df)
        with col4:
            if home_target and len(current_df) > 0:
                t_num = pd.to_numeric(current_df[home_target], errors='coerce').dropna()
                val_str = f"{t_num.mean():.2f} / 20" if not t_num.empty else "N/A"
            else:
                val_str = "N/A"
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-card-title">Avg Final Grade ({home_target if home_target else 'G3'})</div>
                    <div class="stat-card-value">{val_str}</div>
                </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.markdown("### 📋 Preview of Active Dataset")
        st.dataframe(current_df.head(10), use_container_width=True)
        
        # System Features Card Grid
        st.write("")
        st.markdown("### 🚀 Available Modules")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("""
            **📁 Upload Custom Dataset**
            - Support for CSV and Excel (.xlsx) files.
            - Auto-detect delimiters & schema.
            - Quality checks: nulls, duplicates, outliers.
            """)
        with c2:
            st.info("""
            **📊 Dynamic Exploratory Analysis**
            - Interactive multi-parameter filters.
            - Grade distribution, study time & school analysis.
            - Interactive Plotly correlation heatmaps.
            """)
        with c3:
            st.info("""
            **🎯 Grade Prediction System**
            - Individual student grade forecaster.
            - Batch CSV prediction with instant export.
            - Trained model evaluation benchmark.
            """)
    else:
        st.warning("⚠️ No dataset loaded. Please check dataset folder or upload a custom dataset.")


# =========================================================
# 2. UPLOAD STUDENT DATA
# =========================================================
elif menu == "Upload Student Data":
    st.markdown("""
        <div class="header-box">
            <h1>📁 Upload Student Dataset</h1>
            <p>Upload your own institutional data (CSV or Excel) to perform dynamic exploratory data analysis.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    > **Validation Requirements:**
    > - Supported formats: `.csv`, `.xlsx`.
    > - The file must contain at least 1 record and 1 column.
    > - For comprehensive academic grade analysis, a target column (such as `G3`) is recommended.
    """)
    
    uploaded_file = st.file_uploader(
        "Select a CSV or Excel file to upload",
        type=['csv', 'xlsx'],
        help="Upload CSV or Excel file containing student records."
    )
    
    if uploaded_file is not None:
        try:
            # Smart Reading Logic
            if uploaded_file.name.endswith('.csv'):
                raw_bytes = uploaded_file.getvalue()
                # Check encoding & delimiter
                try:
                    sample_str = raw_bytes[:4096].decode('utf-8')
                except UnicodeDecodeError:
                    sample_str = raw_bytes[:4096].decode('latin-1')
                    
                first_line = sample_str.split('\n')[0] if '\n' in sample_str else sample_str
                delimiter = ';' if ';' in first_line else ','
                uploaded_file.seek(0)
                uploaded_df = pd.read_csv(uploaded_file, sep=delimiter)
            else:
                uploaded_df = pd.read_excel(uploaded_file)
                
            # Rigorous Validation: Check if empty
            if uploaded_df is None or uploaded_df.empty or uploaded_df.shape[0] == 0:
                st.error("❌ The uploaded file contains NO records. Please upload a non-empty dataset.")
            elif uploaded_df.shape[1] == 0:
                st.error("❌ The uploaded file contains NO columns. Please upload a valid table structure.")
            else:
                # Save into session state
                st.session_state.uploaded_data = uploaded_df
                st.session_state.uploaded_filename = uploaded_file.name
                
                st.success(f"✅ File **{uploaded_file.name}** uploaded and loaded successfully! All analysis modules now use this data.")
                
                # Metadata Summary
                st.markdown("### 📊 Dataset Overview")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Filename", uploaded_file.name)
                m2.metric("Total Rows", f"{uploaded_df.shape[0]:,}")
                m3.metric("Total Columns", uploaded_df.shape[1])
                m4.metric("Duplicate Rows", uploaded_df.duplicated().sum())
                
                # Column Names
                with st.expander("🔍 View All Column Names", expanded=False):
                    st.write(", ".join([f"`{col}`" for col in uploaded_df.columns]))
                
                # First 5 Records
                st.markdown("### 📋 First 5 Records")
                st.dataframe(uploaded_df.head(5), use_container_width=True)
                
                # Data Types and Missing Values Table
                st.markdown("### 🧬 Column Schema & Missing Values")
                null_counts = uploaded_df.isnull().sum()
                null_pcts = (null_counts / len(uploaded_df)) * 100
                dtypes = uploaded_df.dtypes.astype(str)
                unique_counts = uploaded_df.nunique()
                
                schema_df = pd.DataFrame({
                    "Column Name": uploaded_df.columns,
                    "Data Type": dtypes,
                    "Missing Values": null_counts,
                    "Missing (%)": null_pcts.map(lambda x: f"{x:.2f}%"),
                    "Unique Values": unique_counts
                }).reset_index(drop=True)
                
                st.dataframe(schema_df, use_container_width=True)
                
                # Numerical Summary & Outlier Detection
                st.markdown("### 📈 Numerical Summary & Outlier Inspection")
                num_df = uploaded_df.select_dtypes(include=np.number)
                if not num_df.empty:
                    desc_df = num_df.describe().T
                    # Calculate potential outliers count per column
                    outlier_counts = {}
                    for col in num_df.columns:
                        outlier_counts[col] = detect_outliers_iqr(num_df[col].dropna())
                    desc_df["Outliers (IQR 1.5x)"] = pd.Series(outlier_counts)
                    st.dataframe(desc_df, use_container_width=True)
                else:
                    st.info("ℹ️ No numerical columns found in the uploaded file.")
                
                # Target Column Validation Notice
                detected_target = get_target_column(uploaded_df)
                if detected_target:
                    st.success(f"🎯 Target column `{detected_target}` detected! Full exploratory grade distribution and comparative analytics are enabled.")
                else:
                    st.info("ℹ️ **Target column (`G3`) not present:** This dataset contains student attributes for exploratory analysis and batch forecasting. All attribute distributions, missing value checks, duplicate scans, and outlier analyses are fully available. Final-grade statistical summaries and ground-truth comparisons are simply omitted without error.")
                    
        except Exception as e:
            st.error(f"❌ Failed to parse uploaded file: {str(e)}")


# =========================================================
# 3. STUDENT DATA ANALYSIS & INTERACTIVE VISUALIZATIONS
# =========================================================
elif menu == "Student Data Analysis":
    st.markdown("""
        <div class="header-box">
            <h1>📊 Student Data Analysis & Visualizations</h1>
            <p>Dynamically explore student metrics, apply cohort filters, and examine interactive distribution patterns.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f'<div class="dataset-badge">Active Dataset: {current_dataset_label}</div>', unsafe_allow_html=True)
    
    if current_df is None:
        st.error("❌ No dataset available for analysis. Please upload a dataset in 'Upload Student Data'.")
    else:
        # Target column detection (case-insensitive)
        target_col = get_target_column(current_df)
        
        # -----------------------------------------------------
        # Dynamic Multi-Attribute Filters (Sidebar)
        # -----------------------------------------------------
        st.sidebar.markdown("### 🔍 Cohort Filters")
        filtered_df = current_df.copy()
        filters_applied = False
        
        # Filter: School
        if 'school' in filtered_df.columns:
            school_options = filtered_df['school'].dropna().unique().tolist()
            selected_schools = st.sidebar.multiselect("School", school_options, default=None)
            if selected_schools:
                filtered_df = filtered_df[filtered_df['school'].isin(selected_schools)]
                filters_applied = True
                
        # Filter: Gender / Sex
        gender_col = 'sex' if 'sex' in filtered_df.columns else ('gender' if 'gender' in filtered_df.columns else None)
        if gender_col:
            gender_options = filtered_df[gender_col].dropna().unique().tolist()
            selected_genders = st.sidebar.multiselect("Gender", gender_options, default=None)
            if selected_genders:
                filtered_df = filtered_df[filtered_df[gender_col].isin(selected_genders)]
                filters_applied = True
                
        # Filter: Age
        if 'age' in filtered_df.columns:
            age_num = pd.to_numeric(filtered_df['age'], errors='coerce').dropna()
            if not age_num.empty:
                min_age = int(age_num.min())
                max_age = int(age_num.max())
                if min_age < max_age:
                    selected_age_range = st.sidebar.slider("Age Range", min_age, max_age, (min_age, max_age))
                    if selected_age_range != (min_age, max_age):
                        filtered_df = filtered_df[(pd.to_numeric(filtered_df['age'], errors='coerce') >= selected_age_range[0]) & (pd.to_numeric(filtered_df['age'], errors='coerce') <= selected_age_range[1])]
                        filters_applied = True
                    
        # Filter: Study Time
        if 'studytime' in filtered_df.columns:
            study_options = sorted(filtered_df['studytime'].dropna().unique().tolist())
            selected_study = st.sidebar.multiselect("Study Time Level", study_options, default=None)
            if selected_study:
                filtered_df = filtered_df[filtered_df['studytime'].isin(selected_study)]
                filters_applied = True
                
        # Filter: Previous Grades (G1, G2)
        if 'G1' in filtered_df.columns:
            g1_num = pd.to_numeric(filtered_df['G1'], errors='coerce').dropna()
            if not g1_num.empty:
                min_g1, max_g1 = int(g1_num.min()), int(g1_num.max())
                if min_g1 < max_g1:
                    g1_range = st.sidebar.slider("First Period Grade (G1)", min_g1, max_g1, (min_g1, max_g1))
                    if g1_range != (min_g1, max_g1):
                        filtered_df = filtered_df[(pd.to_numeric(filtered_df['G1'], errors='coerce') >= g1_range[0]) & (pd.to_numeric(filtered_df['G1'], errors='coerce') <= g1_range[1])]
                        filters_applied = True
                    
        if filters_applied:
            st.info(f"🔍 Cohort filters active: Displaying **{len(filtered_df)}** of **{len(current_df)}** records.")
            
        # -----------------------------------------------------
        # Dynamic Statistics Dashboard
        # -----------------------------------------------------
        st.markdown("### 📌 Summary Performance Metrics")
        
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        stat_col1.metric("Total Cohort Records", f"{len(filtered_df):,}")
        stat_col2.metric("Missing Values", filtered_df.isnull().sum().sum())
        stat_col3.metric("Duplicate Records", filtered_df.duplicated().sum())
        
        if target_col and len(filtered_df) > 0:
            target_series = pd.to_numeric(filtered_df[target_col], errors='coerce').dropna()
            if not target_series.empty:
                stat_col4.metric(f"Avg Final Grade ({target_col})", f"{target_series.mean():.2f}")
                
                # Detailed Grade Metrics
                st.markdown(f"#### 🎯 Target Grade Analysis (`{target_col}`)")
                g_col1, g_col2, g_col3, g_col4 = st.columns(4)
                g_col1.metric("Median Grade", f"{target_series.median():.2f}")
                g_col2.metric("Highest Grade", f"{target_series.max():.2f}")
                g_col3.metric("Lowest Grade", f"{target_series.min():.2f}")
                g_col4.metric("Standard Deviation", f"{target_series.std():.2f}")
            else:
                stat_col4.metric("Avg Final Grade", "No Numeric Data")
        else:
            stat_col4.metric("Avg Final Grade", "N/A (No G3)")
            st.info("ℹ️ **Final Grade (`G3`) Analysis Note:** Target grade column is not present in this uploaded dataset. Full cohort attribute breakdowns, demographic filters, missing values, duplicates, and correlation heatmap are actively analyzed below. To predict final grades for these records, navigate to **Batch Prediction**.")
        
        # -----------------------------------------------------
        # Interactive Visualizations (Plotly)
        # -----------------------------------------------------
        st.markdown("---")
        st.markdown("### 📈 Interactive Visualizations")
        
        if len(filtered_df) == 0:
            st.error("⚠️ No records match the active filter criteria. Please adjust your filters.")
        else:
            if target_col:
                # Row 1: Target Grade Distribution & Study Time Boxplot
                row1_c1, row1_c2 = st.columns(2)
                with row1_c1:
                    fig_dist = px.histogram(
                        filtered_df,
                        x=target_col,
                        nbins=20,
                        marginal="box",
                        title=f"Final Grade ({target_col}) Distribution",
                        color_discrete_sequence=['#2563eb']
                    )
                    fig_dist.update_layout(bargap=0.08, template="plotly_white")
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                with row1_c2:
                    if 'studytime' in filtered_df.columns:
                        fig_study = px.box(
                            filtered_df,
                            x='studytime',
                            y=target_col,
                            points="all",
                            title=f"Study Time vs Final Grade ({target_col})",
                            color='studytime',
                            color_discrete_sequence=px.colors.qualitative.Prism
                        )
                        fig_study.update_layout(template="plotly_white", showlegend=False)
                        st.plotly_chart(fig_study, use_container_width=True)
                    else:
                        st.info("ℹ️ `studytime` column not found for Study Time boxplot.")

                # Row 2: Previous Grade vs Final Grade & Gender/School Breakdown
                row2_c1, row2_c2 = st.columns(2)
                with row2_c1:
                    if 'G2' in filtered_df.columns:
                        color_param = 'G1' if 'G1' in filtered_df.columns else None
                        try:
                            fig_prev = px.scatter(
                                filtered_df,
                                x='G2',
                                y=target_col,
                                color=color_param,
                                trendline="ols" if len(filtered_df) > 5 else None,
                                title=f"Previous Grade (G2) vs Final Grade ({target_col})",
                                color_continuous_scale="Blues"
                            )
                        except Exception:
                            # Graceful fallback without trendline if statistical calculation encounters any environment issue
                            fig_prev = px.scatter(
                                filtered_df,
                                x='G2',
                                y=target_col,
                                color=color_param,
                                title=f"Previous Grade (G2) vs Final Grade ({target_col})",
                                color_continuous_scale="Blues"
                            )
                        fig_prev.update_layout(template="plotly_white")
                        st.plotly_chart(fig_prev, use_container_width=True)
                    else:
                        st.info("ℹ️ `G2` column not found for previous grade correlation.")
                        
                with row2_c2:
                    if gender_col in filtered_df.columns:
                        fig_sex = px.violin(
                            filtered_df,
                            x=gender_col,
                            y=target_col,
                            box=True,
                            points="all",
                            title=f"Grade Distribution by Gender ({gender_col})",
                            color=gender_col,
                            color_discrete_sequence=['#3b82f6', '#ec4899']
                        )
                        fig_sex.update_layout(template="plotly_white")
                        st.plotly_chart(fig_sex, use_container_width=True)
                    elif 'school' in filtered_df.columns:
                        fig_school = px.histogram(
                            filtered_df,
                            x=target_col,
                            color="school",
                            barmode="overlay",
                            title=f"Grade Distribution by School"
                        )
                        fig_school.update_layout(template="plotly_white")
                        st.plotly_chart(fig_school, use_container_width=True)
            else:
                # When target_col is not present: Provide rich attribute distribution visualizations
                st.markdown("#### 📊 Student Demographic & Attribute Distributions")
                c_a, c_b = st.columns(2)
                with c_a:
                    if 'age' in filtered_df.columns:
                        fig_age = px.histogram(
                            filtered_df,
                            x='age',
                            nbins=10,
                            title="Age Distribution of Students",
                            color_discrete_sequence=['#2563eb']
                        )
                        fig_age.update_layout(template="plotly_white", bargap=0.08)
                        st.plotly_chart(fig_age, use_container_width=True)
                with c_b:
                    if 'studytime' in filtered_df.columns:
                        fig_st = px.histogram(
                            filtered_df,
                            x='studytime',
                            title="Study Time Distribution (Levels 1 to 4)",
                            color_discrete_sequence=['#10b981']
                        )
                        fig_st.update_layout(template="plotly_white", bargap=0.1)
                        st.plotly_chart(fig_st, use_container_width=True)
                        
                c_c, c_d = st.columns(2)
                with c_c:
                    if gender_col and gender_col in filtered_df.columns:
                        fig_gen = px.pie(
                            filtered_df,
                            names=gender_col,
                            title="Gender Distribution",
                            color_discrete_sequence=['#3b82f6', '#ec4899']
                        )
                        fig_gen.update_layout(template="plotly_white")
                        st.plotly_chart(fig_gen, use_container_width=True)
                with c_d:
                    if 'school' in filtered_df.columns:
                        fig_sch = px.histogram(
                            filtered_df,
                            x='school',
                            title="School Enrollment Distribution",
                            color_discrete_sequence=['#6366f1']
                        )
                        fig_sch.update_layout(template="plotly_white", bargap=0.1)
                        st.plotly_chart(fig_sch, use_container_width=True)

            # Row 4: Correlation Heatmap for Numerical Columns
            st.markdown("#### 🌡️ Numerical Correlation Heatmap")
            num_cols_df = filtered_df.select_dtypes(include=np.number)
            if num_cols_df.shape[1] > 1:
                corr_matrix = num_cols_df.corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    title="Correlation Matrix of Numerical Attributes",
                    aspect="auto"
                )
                fig_corr.update_layout(template="plotly_white")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("ℹ️ At least 2 numerical features are required to compute a correlation matrix.")

            # Row 5: Missing-Value Visualization
            st.markdown("#### 🔍 Missing Values Analysis")
            missing_series = filtered_df.isnull().sum()
            missing_chart_df = pd.DataFrame({
                "Feature": missing_series.index,
                "Missing Count": missing_series.values
            })
            if missing_chart_df['Missing Count'].sum() == 0:
                st.success("🎉 Excellent! Zero missing values detected across all features.")
            else:
                fig_missing = px.bar(
                    missing_chart_df,
                    x="Feature",
                    y="Missing Count",
                    title="Missing Values per Feature",
                    color="Missing Count",
                    color_continuous_scale="Reds"
                )
                fig_missing.update_layout(template="plotly_white")
                st.plotly_chart(fig_missing, use_container_width=True)
                
            # Filtered Data Table View
            st.markdown("### 📋 Filtered Cohort Data View")
            st.dataframe(filtered_df, use_container_width=True)
            
            # Download Filtered Data Button
            csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Filtered Data as CSV",
                data=csv_filtered,
                file_name="filtered_student_data.csv",
                mime="text/csv"
            )


# =========================================================
# 4. INDIVIDUAL STUDENT PREDICTION
# =========================================================
elif menu == "Predict Student Performance":
    st.markdown("""
        <div class="header-box">
            <h1>🎯 Individual Student Performance Prediction</h1>
            <p>Predict academic final grade (G3) using the genuine pre-trained Machine Learning model pipeline.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("""
            ❌ **Trained Model Pipeline Missing!**
            The required model file `Model/best_model.pkl` could not be located.
            
            **How to train and save the model:**
            1. Open your terminal in the project directory.
            2. Run: `python src/train_model.py`
            3. Once training finishes, reload this page.
        """)
    elif default_df is None:
        st.error("❌ The base training dataset `Dataset/student-mat.csv` is missing. Unable to verify expected feature schema.")
    else:
        # Features used by the model
        model_features_df = default_df.drop(columns=['G1', 'G2', 'G3'], errors='ignore')
        expected_cols = model_features_df.columns.tolist()
        
        st.markdown("""
        > **Model Input Schema:**
        > The prediction model was trained without `G1` and `G2` to prevent data leakage and enable early-semester intervention.
        > Inputs strictly match the authentic features from the UCI dataset. No artificial or imaginary fields are included.
        """)
        
        with st.form("single_prediction_form"):
            st.subheader("📝 Enter Student Attributes")
            
            # Partition features into meaningful groups
            academic_lifestyle_cols = ['studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet']
            family_personal_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'romantic']
            health_habits_cols = ['famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'traveltime']
            
            # Any remaining feature in expected_cols not covered by the 3 groups above
            covered_cols = set(academic_lifestyle_cols + family_personal_cols + health_habits_cols)
            other_expected_cols = [c for c in expected_cols if c not in covered_cols]
            
            input_dict = {}
            
            def render_feature_input(col, container):
                """Helper to render appropriate input control for a given feature."""
                series = model_features_df[col]
                if pd.api.types.is_numeric_dtype(series):
                    c_clean = pd.to_numeric(series, errors='coerce').dropna()
                    c_min = int(c_clean.min()) if not c_clean.empty else 0
                    c_max = int(c_clean.max()) if not c_clean.empty else 100
                    c_med = int(c_clean.median()) if not c_clean.empty else 0
                    if c_min == c_max:
                        c_max = c_min + 1
                    return container.number_input(
                        f"`{col}`",
                        min_value=c_min,
                        max_value=c_max,
                        value=c_med,
                        key=f"inp_{col}",
                        help=f"Numerical range: {c_min} to {c_max}"
                    )
                else:
                    # Categorical feature: extract genuine unique categories from dataset
                    choices = sorted([str(x) for x in series.dropna().unique().tolist()])
                    return container.selectbox(
                        f"`{col}`",
                        choices,
                        key=f"inp_{col}",
                        help=f"Categorical values: {', '.join(choices)}"
                    )

            # Section 1: Demographics & Family
            st.markdown("##### 👤 Demographic & Family Background")
            c1, c2, c3 = st.columns(3)
            col_list_1 = [col for col in family_personal_cols if col in expected_cols]
            for i, col in enumerate(col_list_1):
                col_container = [c1, c2, c3][i % 3]
                input_dict[col] = render_feature_input(col, col_container)
            
            # Section 2: Academics & Support (Includes nursery)
            st.markdown("##### 📚 Academic History & Support")
            c4, c5, c6 = st.columns(3)
            col_list_2 = [col for col in academic_lifestyle_cols if col in expected_cols]
            for i, col in enumerate(col_list_2):
                col_container = [c4, c5, c6][i % 3]
                input_dict[col] = render_feature_input(col, col_container)
                    
            # Section 3: Health, Habits & Social
            st.markdown("##### 🏃 Health, Lifestyle & Habits")
            c7, c8, c9 = st.columns(3)
            col_list_3 = [col for col in health_habits_cols if col in expected_cols]
            for i, col in enumerate(col_list_3):
                col_container = [c7, c8, c9][i % 3]
                input_dict[col] = render_feature_input(col, col_container)
                
            # Section 4: Other Attributes (if any)
            if other_expected_cols:
                st.markdown("##### 🔍 Additional Attributes")
                co1, co2, co3 = st.columns(3)
                for i, col in enumerate(other_expected_cols):
                    col_container = [co1, co2, co3][i % 3]
                    input_dict[col] = render_feature_input(col, col_container)
                    
            submit_btn = st.form_submit_button("🔮 Predict Final Grade")
            
        if submit_btn:
            # Rigorous Feature Pre-Validation
            missing_form_features = [col for col in expected_cols if col not in input_dict]
            if missing_form_features:
                st.error(f"❌ Input Incomplete! Missing required features: {', '.join(missing_form_features)}")
            else:
                try:
                    # Prepare single-row DataFrame strictly aligned with trained feature sequence
                    input_df = pd.DataFrame([input_dict])[expected_cols]
                    predicted_val = model.predict(input_df)[0]
                    
                    # Format score display (standard Portuguese grade 0-20 scale)
                    clipped_val = max(0.0, min(20.0, float(predicted_val)))
                    
                    st.markdown("---")
                    st.markdown("### 🏆 Prediction Result")
                    res_col1, res_col2 = st.columns([1, 2])
                    with res_col1:
                        st.metric("Predicted Final Grade (0-20 Scale)", f"{clipped_val:.2f} / 20")
                        if clipped_val >= 14:
                            st.success("🌟 Performance Category: Good to Excellent (14 - 20)")
                        elif clipped_val >= 10:
                            st.info("👍 Performance Category: Satisfactory / Passing (10 - 13.9)")
                        else:
                            st.error("⚠️ Performance Category: At-Risk / Needs Intervention (< 10)")
                            
                    with res_col2:
                        # Visual Gauge
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=clipped_val,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Predicted G3 Grade", 'font': {'size': 20}},
                            gauge={
                                'axis': {'range': [0, 20], 'tickwidth': 1, 'tickcolor': "darkblue"},
                                'bar': {'color': "#2563eb"},
                                'steps': [
                                    {'range': [0, 9.9], 'color': "#fee2e2"},
                                    {'range': [10, 13.9], 'color': "#fef3c7"},
                                    {'range': [14, 20], 'color': "#dcfce7"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 9.9
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                except Exception as ex:
                    st.error(f"❌ Prediction failed: {str(ex)}")


# =========================================================
# 5. BATCH PREDICTION
# =========================================================
elif menu == "Batch Prediction":
    st.markdown("""
        <div class="header-box">
            <h1>📑 Batch CSV Prediction</h1>
            <p>Predict academic grades for multiple students simultaneously via CSV upload with instant validation.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if model is None:
        st.error("❌ Trained model is missing. Please run `python src/train_model.py` first.")
    elif default_df is None:
        st.error("❌ Reference dataset missing. Unable to identify expected model features.")
    else:
        expected_cols = default_df.drop(columns=['G1', 'G2', 'G3'], errors='ignore').columns.tolist()
        
        st.markdown(f"""
        ### 📋 Required Batch Schema
        The uploaded CSV must contain the following **{len(expected_cols)}** features:
        `{', '.join(expected_cols)}`
        
        *(Note: If the uploaded file contains columns `G1`, `G2`, or `G3`, they will be safely set aside during prediction to prevent target contamination).*
        """)
        
        # Download Sample Template Button
        sample_batch_df = default_df.drop(columns=['G1', 'G2', 'G3'], errors='ignore').head(5)
        sample_csv = sample_batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample Batch Template (CSV)",
            data=sample_csv,
            file_name="sample_batch_template.csv",
            mime="text/csv",
            help="Download a pre-formatted template with 5 real student sample rows."
        )
        
        st.write("")
        batch_file = st.file_uploader(
            "Upload Batch CSV file",
            type=['csv'],
            help="Upload a CSV with student attributes matching the required feature set."
        )
        
        if batch_file is not None:
            try:
                raw_bytes = batch_file.getvalue()
                try:
                    sample_str = raw_bytes[:4096].decode('utf-8')
                except UnicodeDecodeError:
                    sample_str = raw_bytes[:4096].decode('latin-1')
                    
                first_line = sample_str.split('\n')[0] if '\n' in sample_str else sample_str
                delimiter = ';' if ';' in first_line else ','
                batch_file.seek(0)
                batch_df = pd.read_csv(batch_file, sep=delimiter)
                
                if batch_df.empty or len(batch_df) == 0:
                    st.error("❌ The uploaded batch CSV contains NO records.")
                else:
                    # Schema check
                    missing_features = [col for col in expected_cols if col not in batch_df.columns]
                    
                    if missing_features:
                        st.error(f"❌ Incompatible format! The uploaded file is missing **{len(missing_features)}** required columns:")
                        st.write(", ".join([f"`{c}`" for c in missing_features]))
                    else:
                        st.success(f"✅ Schema validated successfully! Processing **{len(batch_df)}** student records...")
                        
                        # Extract exact features for model
                        X_input = batch_df[expected_cols].copy()
                        
                        with st.spinner("Calculating predictions using ML pipeline..."):
                            predictions = model.predict(X_input)
                            # Clip predictions to sensible academic range [0, 20]
                            predictions_clipped = np.clip(predictions, 0.0, 20.0).round(2)
                            
                        # Preserve original records & add predicted_final_grade
                        results_df = batch_df.copy()
                        results_df['predicted_final_grade'] = predictions_clipped
                        
                        st.markdown("### 📊 Batch Prediction Results")
                        r_c1, r_c2, r_c3 = st.columns(3)
                        r_c1.metric("Total Predicted Students", len(results_df))
                        r_c2.metric("Mean Predicted Grade", f"{predictions_clipped.mean():.2f}")
                        r_c3.metric("Students At-Risk (<10)", int((predictions_clipped < 10).sum()))
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download Output
                        result_csv = results_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions Result as CSV",
                            data=result_csv,
                            file_name="student_performance_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Visual distribution of batch predictions
                        fig_batch_hist = px.histogram(
                            results_df,
                            x="predicted_final_grade",
                            nbins=15,
                            title="Distribution of Batch Predicted Final Grades",
                            color_discrete_sequence=['#2563eb']
                        )
                        fig_batch_hist.update_layout(bargap=0.08, template="plotly_white")
                        st.plotly_chart(fig_batch_hist, use_container_width=True)
                        
            except Exception as e:
                st.error(f"❌ Error processing batch file: {str(e)}")


# =========================================================
# 6. MODEL PERFORMANCE EVALUATION DASHBOARD
# =========================================================
elif menu == "Model Performance":
    st.markdown("""
        <div class="header-box">
            <h1>📈 Machine Learning Model Performance Dashboard</h1>
            <p>Benchmark comparison of trained models evaluated on the held-out test dataset (20% split).</p>
        </div>
    """, unsafe_allow_html=True)
    
    results_path = os.path.join("Output", "model_results.csv")
    alt_results_path = os.path.join("..", "Output", "model_results.csv")
    
    found_path = results_path if os.path.exists(results_path) else (alt_results_path if os.path.exists(alt_results_path) else None)
    
    if found_path is not None:
        metrics_df = pd.read_csv(found_path)
        
        st.markdown("""
        > **Evaluation Methodology:**
        > Metrics were computed strictly on the **unseen 20% test dataset** (`random_state=42`) using Scikit-Learn `Pipeline` with Standard Scaling and One-Hot Encoding. No data leakage from `G1` or `G2` was permitted.
        """)
        
        st.markdown("### 🏆 Model Comparison Benchmark")
        # Format table for high clarity
        st.dataframe(
            metrics_df.style.highlight_max(subset=['R2 Score'], color='#dcfce7')
                           .highlight_min(subset=['RMSE', 'MAE', 'MSE'], color='#dcfce7'),
            use_container_width=True
        )
        
        # Model metric charts
        c1, c2 = st.columns(2)
        with c1:
            fig_r2 = px.bar(
                metrics_df,
                x='Model',
                y='R2 Score',
                color='Model',
                text_auto='.4f',
                title="R² Score Comparison (Higher is Better)",
                color_discrete_sequence=['#2563eb', '#3b82f6', '#1d4ed8']
            )
            fig_r2.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_r2, use_container_width=True)
            
        with c2:
            fig_rmse = px.bar(
                metrics_df,
                x='Model',
                y='RMSE',
                color='Model',
                text_auto='.4f',
                title="Root Mean Squared Error (RMSE) (Lower is Better)",
                color_discrete_sequence=['#f97316', '#ea580c', '#c2410c']
            )
            fig_rmse.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_rmse, use_container_width=True)
            
        # MAE and MSE comparison
        c3, c4 = st.columns(2)
        with c3:
            fig_mae = px.bar(
                metrics_df,
                x='Model',
                y='MAE',
                color='Model',
                text_auto='.4f',
                title="Mean Absolute Error (MAE)",
                color_discrete_sequence=['#06b6d4', '#0891b2', '#0e7490']
            )
            fig_mae.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_mae, use_container_width=True)
            
        with c4:
            fig_mse = px.bar(
                metrics_df,
                x='Model',
                y='MSE',
                color='Model',
                text_auto='.4f',
                title="Mean Squared Error (MSE)",
                color_discrete_sequence=['#8b5cf6', '#7c3aed', '#6d28d9']
            )
            fig_mse.update_layout(template="plotly_white", showlegend=False)
            st.plotly_chart(fig_mse, use_container_width=True)
            
        # Saved model info
        st.markdown("### 🔍 Production Model Architecture")
        if model is not None:
            pipeline_steps = list(model.named_steps.keys()) if hasattr(model, 'named_steps') else "Custom Pipeline"
            model_algo = model.named_steps['model'].__class__.__name__ if hasattr(model, 'named_steps') and 'model' in model.named_steps else "Gradient Boosting"
            
            st.info(f"""
            - **Selected Deployment Model:** `{model_algo}`
            - **Pipeline Stages:** `{" ➔ ".join(pipeline_steps)}`
            - **Handling Categorical Data:** `OneHotEncoder(handle_unknown='ignore')`
            - **Handling Numerical Data:** `StandardScaler()`
            """)
        else:
            st.warning("⚠️ `Model/best_model.pkl` is not currently loaded in session.")
    else:
        st.error("""
            ❌ **Evaluation metrics file not found (`Output/model_results.csv`).**
            To evaluate the models and generate metrics:
            ```bash
            python src/train_model.py
            ```
        """)
