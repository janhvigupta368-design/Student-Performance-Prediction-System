import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import io

# ---------------------------------------------------------
# Cross-Platform Base Path Resolution (Windows & Streamlit Cloud Linux)
# ---------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent

# ---------------------------------------------------------
# Page Configuration & Global Branding
# ---------------------------------------------------------
st.set_page_config(
    page_title="EduPredict AI | Student Performance Analytics & Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Set native logo in sticky top header
logo_candidates = [
    APP_DIR / "logo.svg",
    ROOT_DIR / "Streamlit_App" / "logo.svg",
    Path("Streamlit_App/logo.svg").resolve()
]
for lp in logo_candidates:
    if lp.is_file():
        st.logo(str(lp), size="large")
        break

# ---------------------------------------------------------
# Blue + White Academic Theme & Top Navbar CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Google Fonts: Manrope (Headings) & Inter (Body) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

    /* Global Background & Text */
    html, body, [data-testid="stAppViewContainer"], .main {
        background-color: #F8FAFC !important;
        color: #172B4D !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    }

    /* Completely Remove Sidebar & Chevrons */
    [data-testid="stSidebar"], 
    [data-testid="stSidebarCollapseButton"],
    button[data-testid="stSidebarHeaderChevron"] {
        display: none !important;
    }

    /* Sticky Permanent White Top Header Bar */
    header[data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        border-bottom: 1px solid #E2E8F0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
        position: sticky !important;
        top: 0 !important;
        z-index: 999 !important;
        padding: 0 1.5rem !important;
    }

    /* Top Navigation Links in Header */
    header[data-testid="stHeader"] nav {
        display: flex !important;
        align-items: center !important;
        gap: 4px !important;
        overflow-x: auto !important;
        white-space: nowrap !important;
    }

    header[data-testid="stHeader"] nav a {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: #173B67 !important;
        padding: 6px 14px !important;
        border-radius: 6px !important;
        transition: all 0.15s ease !important;
        text-decoration: none !important;
    }

    header[data-testid="stHeader"] nav a:hover {
        background-color: #EFF6FF !important;
        color: #2563EB !important;
    }

    header[data-testid="stHeader"] nav a[aria-current="page"] {
        background-color: #EFF6FF !important;
        color: #2563EB !important;
        font-weight: 700 !important;
    }

    /* Headings font */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Manrope', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #173B67 !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
    }

    p, span, label, div {
        color: #172B4D;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #64748B !important;
    }

    /* Main Container Full Width Spacing */
    .main .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: clamp(1rem, 2.5vw, 2.5rem) !important;
        padding-right: clamp(1rem, 2.5vw, 2.5rem) !important;
        max-width: 1440px !important;
    }

    /* Clean Compact Dashboard Header Card */
    .dashboard-header-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        padding: 20px 24px;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    .dashboard-header-card h2 {
        color: #173B67 !important;
        margin: 0 !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        line-height: 1.25 !important;
    }
    .dashboard-header-card p {
        color: #64748B !important;
        margin-top: 6px !important;
        margin-bottom: 0 !important;
        font-size: 0.95rem !important;
        line-height: 1.45 !important;
    }

    /* Dataset Status Card */
    .dataset-status-bar {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        flex-wrap: wrap;
    }
    .dataset-status-text {
        color: #173B67;
        font-size: 0.88rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .dataset-status-badge {
        background-color: #EFF6FF;
        color: #2563EB;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* Metric Cards */
    .stat-card {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 16px 20px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
        border-left: 4px solid #2563EB !important;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .stat-card-title {
        color: #64748B !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    .stat-card-value {
        color: #173B67 !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        font-family: 'Manrope', sans-serif !important;
        margin-top: 4px !important;
    }

    /* Streamlit Native Metric Overrides */
    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 14px 18px !important;
        border-left: 4px solid #2563EB !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02) !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 0.78rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    [data-testid="stMetricValue"] {
        color: #173B67 !important;
        font-family: 'Manrope', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1.55rem !important;
    }

    /* White Background Form Card */
    div[data-testid="stForm"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        padding: 24px !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02) !important;
    }

    /* Inputs */
    input, textarea, [data-baseweb="select"], [data-baseweb="input"] {
        background-color: #FFFFFF !important;
        color: #172B4D !important;
        border-color: #E2E8F0 !important;
    }

    /* Buttons */
    .stButton > button, div[data-testid="stFormSubmitButton"] > button {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        border: 1px solid #2563EB !important;
        padding: 8px 20px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(37, 99, 235, 0.15) !important;
    }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #173B67 !important;
        border-color: #173B67 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 8px rgba(23, 59, 103, 0.2) !important;
    }

    /* Download Buttons */
    .stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #173B67 !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 3px solid #2563EB !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 8px 18px !important;
        transition: all 0.2s ease !important;
    }
    .stDownloadButton > button:hover {
        background-color: #EFF6FF !important;
        border-color: #2563EB !important;
        color: #2563EB !important;
    }

    /* Tables & Expanders */
    div[data-testid="stDataFrame"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }

    /* Responsive Grid Breakpoints */
    @media (max-width: 900px) {
        div[data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 12px !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 calc(50% - 12px) !important;
            min-width: calc(50% - 12px) !important;
        }
        div[data-testid="stForm"] div[data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }
    }
    @media (max-width: 600px) {
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
            gap: 10px !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            flex: 1 1 100% !important;
            min-width: 100% !important;
            width: 100% !important;
        }
        .stButton > button, div[data-testid="stFormSubmitButton"] > button, .stDownloadButton > button {
            width: 100% !important;
            min-height: 44px !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Helper Functions, Resource Loaders & Chart Styler
# ---------------------------------------------------------
@st.cache_data
def load_default_data():
    """Load the authentic UCI student dataset using reliable cross-platform paths."""
    candidate_paths = [
        ROOT_DIR / "Dataset" / "student-mat.csv",
        APP_DIR / "Dataset" / "student-mat.csv",
        ROOT_DIR / "student-mat.csv",
        Path("Dataset/student-mat.csv").resolve(),
        Path("student-mat.csv").resolve()
    ]
    for p in candidate_paths:
        if p.is_file():
            for sep in [';', ',']:
                try:
                    df = pd.read_csv(p, sep=sep)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    pass
    return None

@st.cache_resource
def load_trained_model():
    """Load the saved trained ML pipeline model using reliable cross-platform paths."""
    candidate_paths = [
        ROOT_DIR / "Model" / "best_model.pkl",
        APP_DIR / "Model" / "best_model.pkl",
        ROOT_DIR / "best_model.pkl",
        Path("Model/best_model.pkl").resolve(),
        Path("best_model.pkl").resolve()
    ]
    for p in candidate_paths:
        if p.is_file():
            try:
                return joblib.load(p)
            except Exception:
                pass
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

def apply_blue_white_chart(fig, height=350, show_legend=True):
    """Apply consistent clean blue + white academic theme to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=height,
        autosize=True,
        margin=dict(l=15, r=15, t=38, b=25),
        font=dict(family="Inter, sans-serif", color="#172B4D", size=12),
        title_font=dict(family="Manrope, sans-serif", color="#173B67", size=15),
        showlegend=show_legend,
        xaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
        yaxis=dict(gridcolor="#F1F5F9", zerolinecolor="#E2E8F0"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#172B4D")
        ) if show_legend else dict()
    )
    return fig


# ---------------------------------------------------------
# Global State & Dataset Resolution
# ---------------------------------------------------------
default_df = load_default_data()
model = load_trained_model()

if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'uploaded_filename' not in st.session_state:
    st.session_state.uploaded_filename = None

def get_current_dataset():
    """Returns (dataframe, source_label, is_uploaded, filename)."""
    if st.session_state.uploaded_data is not None:
        return st.session_state.uploaded_data, f"User-Uploaded ({st.session_state.uploaded_filename})", True, st.session_state.uploaded_filename
    if default_df is not None:
        return default_df, "Original UCI Dataset (student-mat.csv)", False, "student-mat.csv"
    return None, "No Dataset Available", False, None

def render_dataset_status_bar():
    """Renders a clean, compact dataset status bar with source indicator and metrics."""
    df, label, is_uploaded, filename = get_current_dataset()
    col_status, col_btn = st.columns([4, 1], vertical_alignment="center")
    with col_status:
        if df is not None:
            rec_count = df.shape[0]
            col_count = df.shape[1]
            missing_count = df.isnull().sum().sum()
            source_icon = "📁" if is_uploaded else "🏛️"
            st.markdown(f"""
                <div class="dataset-status-bar">
                    <div class="dataset-status-text">
                        <span>{source_icon} Data Source:</span>
                        <strong>{label}</strong>
                        <span class="dataset-status-badge">{rec_count:,} records</span>
                        <span class="dataset-status-badge">{col_count} features</span>
                        <span class="dataset-status-badge">{missing_count} missing</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="dataset-status-bar" style="border-left-color: #DC2626;">
                    <div class="dataset-status-text">
                        <span>⚠️ Data Source:</span>
                        <strong style="color: #DC2626;">No Dataset Loaded</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    with col_btn:
        if is_uploaded:
            if st.button("🔄 Revert to Default", use_container_width=True, help="Revert to authentic UCI student-mat.csv"):
                st.session_state.uploaded_data = None
                st.session_state.uploaded_filename = None
                st.rerun()


# =========================================================
# 1. PAGE: HOME DASHBOARD
# =========================================================
def page_home_dashboard():
    # Compact Header
    st.markdown("""
        <div class="dashboard-header-card">
            <h2>Student Performance Analytics</h2>
            <p>Explore student data, understand academic patterns, and predict performance using Machine Learning.</p>
        </div>
    """, unsafe_allow_html=True)

    render_dataset_status_bar()
    current_df, source_label, is_uploaded, filename = get_current_dataset()

    if current_df is not None:
        # Four Dynamic Metric Cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-card-title">Total Students</div>
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
                    <div class="stat-card-value">{missing_total:,}</div>
                </div>
            """, unsafe_allow_html=True)
        home_target = get_target_column(current_df)
        with col4:
            if home_target and len(current_df) > 0:
                t_num = pd.to_numeric(current_df[home_target], errors='coerce').dropna()
                val_str = f"{t_num.mean():.2f} / 20" if not t_num.empty else "Not available"
            else:
                val_str = "Not available"
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-card-title">Average Final Grade</div>
                    <div class="stat-card-value">{val_str}</div>
                </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("### 📋 Preview of Active Dataset")
        st.caption(f"Showing top 10 rows of {current_df.shape[0]:,} authentic records from `{filename}`.")
        st.dataframe(current_df.head(10), use_container_width=True)

        with st.expander("🔍 View All Dataset Records", expanded=False):
            st.dataframe(current_df, use_container_width=True)

        # Active Dataset Summary
        st.write("")
        st.markdown("### 🧬 Dataset Summary & Attribute Breakdown")
        num_cols = current_df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = current_df.select_dtypes(exclude=np.number).columns.tolist()
        dup_count = current_df.duplicated().sum()

        sum_c1, sum_c2, sum_c3, sum_c4 = st.columns(4)
        sum_c1.metric("Numerical Features", len(num_cols))
        sum_c2.metric("Categorical Features", len(cat_cols))
        sum_c3.metric("Duplicate Rows", int(dup_count))
        sum_c4.metric("Target (G3) Available", "Yes" if home_target else "No")

        # Quick Modules
        st.write("")
        st.markdown("### 🚀 Available System Modules")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.info("""
            **📁 Upload Custom Data**
            - Support for CSV and Excel (.xlsx, .xls) files.
            - Auto-detect delimiters, encodings & schema.
            - Quality checks: nulls, duplicates, outliers.
            - Keeps uploaded dataset active across pages.
            """)
        with c2:
            st.info("""
            **📊 Dynamic Data Analysis**
            - Interactive multi-parameter cohort filters.
            - Dynamic grade distribution & relationship plots.
            - Resilient exploratory charts if target is missing.
            - Interactive Plotly correlation heatmaps.
            """)
        with c3:
            st.info("""
            **🎯 Grade Prediction System**
            - Individual student grade forecaster.
            - Batch CSV prediction with instant export.
            - Genuine trained model pipeline without fabrication.
            - Benchmark evaluation comparison.
            """)
    else:
        st.error("""
            ❌ **No Active Dataset Available!**
            The original dataset `Dataset/student-mat.csv` could not be located on the server, and no user file has been uploaded yet.
            Please navigate to the **Upload Data** tab to upload a CSV or Excel dataset.
        """)


# =========================================================
# 2. PAGE: UPLOAD STUDENT DATA
# =========================================================
def page_upload_data():
    st.markdown("""
        <div class="dashboard-header-card">
            <h2>📁 Upload Student Dataset</h2>
            <p>Upload institutional student records in CSV or Excel format to perform live analytics and batch forecasts.</p>
        </div>
    """, unsafe_allow_html=True)

    render_dataset_status_bar()

    st.markdown("""
    > **File Upload Guidelines:**
    > - Supported formats: `.csv`, `.xlsx`, `.xls`
    > - Delimiters supported: comma (`,`), semicolon (`;`), tab (`\\t`)
    > - For academic grade forecasting, a target column (`G3`) is recommended.
    > - Once uploaded, the dataset remains active across all navigation pages until you reset it.
    """)

    uploaded_file = st.file_uploader(
        "Select a CSV or Excel file to upload",
        type=['csv', 'xlsx', 'xls'],
        help="Upload CSV or Excel file containing student records."
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_bytes = uploaded_file.getvalue()
                # Encoding detection
                enc = 'utf-8'
                try:
                    sample_str = raw_bytes[:4096].decode('utf-8')
                except UnicodeDecodeError:
                    try:
                        sample_str = raw_bytes[:4096].decode('latin-1')
                        enc = 'latin-1'
                    except Exception:
                        sample_str = raw_bytes[:4096].decode('cp1252', errors='ignore')
                        enc = 'cp1252'

                first_line = sample_str.split('\n')[0] if '\n' in sample_str else sample_str
                if ';' in first_line:
                    sep = ';'
                elif '\t' in first_line:
                    sep = '\t'
                else:
                    sep = ','

                uploaded_file.seek(0)
                new_df = pd.read_csv(uploaded_file, sep=sep, encoding=enc)
            else:
                new_df = pd.read_excel(uploaded_file)

            if new_df is None or new_df.empty or new_df.shape[0] == 0:
                st.error("❌ The uploaded file contains NO records. Please upload a non-empty dataset.")
            elif new_df.shape[1] == 0:
                st.error("❌ The uploaded file contains NO columns. Please upload a valid table structure.")
            else:
                st.session_state.uploaded_data = new_df
                st.session_state.uploaded_filename = uploaded_file.name
                st.success(f"✅ File **{uploaded_file.name}** uploaded successfully! Active across all navigation pages.")
                st.rerun()

        except Exception as e:
            st.error(f"❌ Failed to parse uploaded file: {str(e)}")

    # Display active uploaded dataset if present
    if st.session_state.uploaded_data is not None:
        active_up_df = st.session_state.uploaded_data
        active_up_name = st.session_state.uploaded_filename

        st.markdown(f"### 📊 Active Uploaded Dataset: `{active_up_name}`")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Filename", active_up_name)
        m2.metric("Total Rows", f"{active_up_df.shape[0]:,}")
        m3.metric("Total Columns", active_up_df.shape[1])
        m4.metric("Duplicate Rows", int(active_up_df.duplicated().sum()))

        detected_target = get_target_column(active_up_df)
        if detected_target:
            t_vals = pd.to_numeric(active_up_df[detected_target], errors='coerce').dropna()
            mean_g = f"{t_vals.mean():.2f}" if not t_vals.empty else "N/A"
            st.success(f"🎯 Target column `{detected_target}` detected! (Calculated Average: {mean_g}). Full grade distribution and analysis are enabled.")
        else:
            st.info("ℹ️ **Target column (`G3`) not detected:** This dataset can still be analyzed for demographic distributions, correlations, and used in Batch Prediction.")

        with st.expander("🔍 View All Column Names", expanded=False):
            st.write(", ".join([f"`{col}`" for col in active_up_df.columns]))

        st.markdown("### 📋 Preview (First 5 Records)")
        st.dataframe(active_up_df.head(5), use_container_width=True)

        st.markdown("### 🧬 Column Schema & Missing Values")
        null_counts = active_up_df.isnull().sum()
        null_pcts = (null_counts / len(active_up_df)) * 100
        dtypes = active_up_df.dtypes.astype(str)
        unique_counts = active_up_df.nunique()

        schema_df = pd.DataFrame({
            "Column Name": active_up_df.columns,
            "Data Type": dtypes,
            "Missing Values": null_counts,
            "Missing (%)": null_pcts.map(lambda x: f"{x:.2f}%"),
            "Unique Values": unique_counts
        }).reset_index(drop=True)
        st.dataframe(schema_df, use_container_width=True)

        st.markdown("### 📈 Numerical Summary & Outlier Inspection")
        num_df = active_up_df.select_dtypes(include=np.number)
        if not num_df.empty:
            desc_df = num_df.describe().T
            outlier_counts = {col: detect_outliers_iqr(num_df[col].dropna()) for col in num_df.columns}
            desc_df["Outliers (IQR 1.5x)"] = pd.Series(outlier_counts)
            st.dataframe(desc_df, use_container_width=True)
        else:
            st.info("ℹ️ No numerical columns found in the uploaded file.")

        if st.button("🔄 Clear Upload & Revert to Default Dataset", key="btn_clear_up"):
            st.session_state.uploaded_data = None
            st.session_state.uploaded_filename = None
            st.rerun()

    elif default_df is not None:
        st.info("ℹ️ Currently analyzing the default authentic UCI dataset (`student-mat.csv`). Upload a new file above to analyze your custom records.")


# =========================================================
# 3. PAGE: STUDENT DATA ANALYSIS
# =========================================================
def page_data_analysis():
    st.markdown("""
        <div class="dashboard-header-card">
            <h2>📊 Student Data Analysis & Visualizations</h2>
            <p>Dynamically explore student metrics, filter cohorts, and investigate grade correlations with interactive Plotly charts.</p>
        </div>
    """, unsafe_allow_html=True)

    render_dataset_status_bar()
    current_df, source_label, is_uploaded, filename = get_current_dataset()

    if current_df is None:
        st.error("❌ No dataset available for analysis. Please upload a dataset in 'Upload Data'.")
        return

    target_col = get_target_column(current_df)

    # Clean Dynamic Cohort Filters
    with st.expander("🔍 Cohort Filters (Click to expand/collapse)", expanded=True):
        filtered_df = current_df.copy()
        filters_applied = False

        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            if 'school' in filtered_df.columns:
                school_opts = filtered_df['school'].dropna().unique().tolist()
                sel_schools = st.multiselect("School", school_opts, default=None)
                if sel_schools:
                    filtered_df = filtered_df[filtered_df['school'].isin(sel_schools)]
                    filters_applied = True
            else:
                cat_cols = filtered_df.select_dtypes(exclude=np.number).columns.tolist()
                if len(cat_cols) > 0 and filtered_df[cat_cols[0]].nunique() <= 15:
                    c_col = cat_cols[0]
                    c_opts = filtered_df[c_col].dropna().unique().tolist()
                    sel_c = st.multiselect(f"{c_col}", c_opts, default=None)
                    if sel_c:
                        filtered_df = filtered_df[filtered_df[c_col].isin(sel_c)]
                        filters_applied = True

        with fc2:
            gender_col = 'sex' if 'sex' in filtered_df.columns else ('gender' if 'gender' in filtered_df.columns else None)
            if gender_col:
                gender_opts = filtered_df[gender_col].dropna().unique().tolist()
                sel_genders = st.multiselect("Gender", gender_opts, default=None)
                if sel_genders:
                    filtered_df = filtered_df[filtered_df[gender_col].isin(sel_genders)]
                    filters_applied = True
            else:
                cat_cols = filtered_df.select_dtypes(exclude=np.number).columns.tolist()
                if len(cat_cols) > 1 and filtered_df[cat_cols[1]].nunique() <= 15:
                    c_col2 = cat_cols[1]
                    c_opts2 = filtered_df[c_col2].dropna().unique().tolist()
                    sel_c2 = st.multiselect(f"{c_col2}", c_opts2, default=None)
                    if sel_c2:
                        filtered_df = filtered_df[filtered_df[c_col2].isin(sel_c2)]
                        filters_applied = True

        with fc3:
            if 'age' in filtered_df.columns:
                age_num = pd.to_numeric(filtered_df['age'], errors='coerce').dropna()
                if not age_num.empty and int(age_num.min()) < int(age_num.max()):
                    min_age, max_age = int(age_num.min()), int(age_num.max())
                    sel_age = st.slider("Age Range", min_age, max_age, (min_age, max_age))
                    if sel_age != (min_age, max_age):
                        filtered_df = filtered_df[(pd.to_numeric(filtered_df['age'], errors='coerce') >= sel_age[0]) & 
                                                  (pd.to_numeric(filtered_df['age'], errors='coerce') <= sel_age[1])]
                        filters_applied = True

        fc4, fc5 = st.columns(2)
        with fc4:
            if 'studytime' in filtered_df.columns:
                study_opts = sorted(filtered_df['studytime'].dropna().unique().tolist())
                sel_study = st.multiselect("Study Time Level", study_opts, default=None)
                if sel_study:
                    filtered_df = filtered_df[filtered_df['studytime'].isin(sel_study)]
                    filters_applied = True
        with fc5:
            if 'G1' in filtered_df.columns:
                g1_num = pd.to_numeric(filtered_df['G1'], errors='coerce').dropna()
                if not g1_num.empty and int(g1_num.min()) < int(g1_num.max()):
                    min_g1, max_g1 = int(g1_num.min()), int(g1_num.max())
                    g1_range = st.slider("First Period Grade (G1)", min_g1, max_g1, (min_g1, max_g1))
                    if g1_range != (min_g1, max_g1):
                        filtered_df = filtered_df[(pd.to_numeric(filtered_df['G1'], errors='coerce') >= g1_range[0]) & 
                                                  (pd.to_numeric(filtered_df['G1'], errors='coerce') <= g1_range[1])]
                        filters_applied = True

        if filters_applied:
            st.info(f"🔍 Active Filter: Showing **{len(filtered_df):,}** of **{len(current_df):,}** students.")

    # Performance Metrics
    st.write("")
    st.markdown("### 📌 Cohort Performance Metrics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    stat_col1.metric("Cohort Students", f"{len(filtered_df):,}")
    stat_col2.metric("Missing Values", f"{filtered_df.isnull().sum().sum():,}")
    stat_col3.metric("Duplicate Records", int(filtered_df.duplicated().sum()))

    if target_col and len(filtered_df) > 0:
        target_series = pd.to_numeric(filtered_df[target_col], errors='coerce').dropna()
        if not target_series.empty:
            stat_col4.metric(f"Avg Final Grade ({target_col})", f"{target_series.mean():.2f}")
            g_c1, g_c2, g_c3, g_c4 = st.columns(4)
            g_c1.metric("Median Grade", f"{target_series.median():.2f}")
            g_c2.metric("Highest Grade", f"{target_series.max():.2f}")
            g_c3.metric("Lowest Grade", f"{target_series.min():.2f}")
            g_c4.metric("Standard Deviation", f"{target_series.std():.2f}")
        else:
            stat_col4.metric("Avg Final Grade", "No Numeric Data")
    else:
        stat_col4.metric("Avg Final Grade", "Not available")

    # Interactive Plotly Charts
    st.markdown("---")
    st.markdown("### 📈 Visualizations")

    if len(filtered_df) == 0:
        st.error("⚠️ No records match the active filter criteria. Please adjust your filters above.")
    else:
        if target_col:
            r1_c1, r1_c2 = st.columns(2)
            with r1_c1:
                fig_dist = px.histogram(
                    filtered_df,
                    x=target_col,
                    nbins=20,
                    marginal="box",
                    title=f"Final Grade ({target_col}) Distribution",
                    color_discrete_sequence=['#2563EB']
                )
                fig_dist.update_layout(bargap=0.08)
                apply_blue_white_chart(fig_dist, height=350, show_legend=False)
                st.plotly_chart(fig_dist, use_container_width=True, config={'responsive': True, 'displayModeBar': False})

            with r1_c2:
                if 'studytime' in filtered_df.columns:
                    fig_study = px.box(
                        filtered_df,
                        x='studytime',
                        y=target_col,
                        points="all",
                        title=f"Study Time vs Final Grade ({target_col})",
                        color='studytime',
                        color_discrete_sequence=['#173B67', '#2563EB', '#38BDF8', '#0284C7']
                    )
                    apply_blue_white_chart(fig_study, height=350, show_legend=False)
                    st.plotly_chart(fig_study, use_container_width=True, config={'responsive': True, 'displayModeBar': False})
                else:
                    num_sub = [c for c in filtered_df.select_dtypes(include=np.number).columns if c != target_col]
                    if num_sub:
                        fig_alt = px.scatter(
                            filtered_df,
                            x=num_sub[0],
                            y=target_col,
                            title=f"{num_sub[0]} vs Final Grade ({target_col})",
                            color_discrete_sequence=['#2563EB']
                        )
                        apply_blue_white_chart(fig_alt, height=350, show_legend=False)
                        st.plotly_chart(fig_alt, use_container_width=True, config={'responsive': True, 'displayModeBar': False})
                    else:
                        st.info("ℹ️ No secondary numerical feature available for grade comparison.")

            r2_c1, r2_c2 = st.columns(2)
            with r2_c1:
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
                        fig_prev = px.scatter(
                            filtered_df,
                            x='G2',
                            y=target_col,
                            color=color_param,
                            title=f"Previous Grade (G2) vs Final Grade ({target_col})",
                            color_continuous_scale="Blues"
                        )
                    apply_blue_white_chart(fig_prev, height=350, show_legend=True)
                    st.plotly_chart(fig_prev, use_container_width=True, config={'responsive': True, 'displayModeBar': False})
                else:
                    st.info("ℹ️ `G2` column not found in this dataset.")

            with r2_c2:
                gender_col = 'sex' if 'sex' in filtered_df.columns else ('gender' if 'gender' in filtered_df.columns else None)
                if gender_col:
                    fig_sex = px.violin(
                        filtered_df,
                        x=gender_col,
                        y=target_col,
                        box=True,
                        points="all",
                        title=f"Grade Distribution by Gender ({gender_col})",
                        color=gender_col,
                        color_discrete_sequence=['#2563EB', '#38BDF8']
                    )
                    apply_blue_white_chart(fig_sex, height=350, show_legend=False)
                    st.plotly_chart(fig_sex, use_container_width=True, config={'responsive': True, 'displayModeBar': False})
                elif 'school' in filtered_df.columns:
                    fig_school = px.histogram(
                        filtered_df,
                        x=target_col,
                        color="school",
                        barmode="overlay",
                        title="Grade Distribution by School",
                        color_discrete_sequence=['#173B67', '#2563EB']
                    )
                    apply_blue_white_chart(fig_school, height=350, show_legend=True)
                    st.plotly_chart(fig_school, use_container_width=True, config={'responsive': True, 'displayModeBar': False})
        else:
            # Resilient exploratory charts when G3 is missing
            st.info(f"ℹ️ **Final-grade analysis is unavailable for this dataset ('{filename}') because the target column ('G3') was not detected.** Below are dynamic exploratory distributions of available attributes.")

            num_cols_avail = filtered_df.select_dtypes(include=np.number).columns.tolist()
            cat_cols_avail = filtered_df.select_dtypes(exclude=np.number).columns.tolist()

            c_a, c_b = st.columns(2)
            with c_a:
                if len(num_cols_avail) > 0:
                    fig_n1 = px.histogram(
                        filtered_df,
                        x=num_cols_avail[0],
                        nbins=15,
                        title=f"Distribution of `{num_cols_avail[0]}`",
                        color_discrete_sequence=['#2563EB']
                    )
                    fig_n1.update_layout(bargap=0.08)
                    apply_blue_white_chart(fig_n1, height=340, show_legend=False)
                    st.plotly_chart(fig_n1, use_container_width=True, config={'responsive': True, 'displayModeBar': False})
                else:
                    st.info("ℹ️ No numerical columns available to plot.")

            with c_b:
                if len(num_cols_avail) > 1:
                    fig_n2 = px.histogram(
                        filtered_df,
                        x=num_cols_avail[1],
                        nbins=15,
                        title=f"Distribution of `{num_cols_avail[1]}`",
                        color_discrete_sequence=['#38BDF8']
                    )
                    fig_n2.update_layout(bargap=0.08)
                    apply_blue_white_chart(fig_n2, height=340, show_legend=False)
                    st.plotly_chart(fig_n2, use_container_width=True, config={'responsive': True, 'displayModeBar': False})
                elif len(cat_cols_avail) > 0:
                    fig_cat = px.histogram(
                        filtered_df,
                        x=cat_cols_avail[0],
                        title=f"Category Distribution of `{cat_cols_avail[0]}`",
                        color_discrete_sequence=['#173B67']
                    )
                    apply_blue_white_chart(fig_cat, height=340, show_legend=False)
                    st.plotly_chart(fig_cat, use_container_width=True, config={'responsive': True, 'displayModeBar': False})

        # Numerical Correlation Heatmap
        st.markdown("#### 🌡️ Numerical Correlation Heatmap")
        num_cols_df = filtered_df.select_dtypes(include=np.number)
        if num_cols_df.shape[1] > 1:
            corr_matrix = num_cols_df.corr()
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="Blues",
                zmin=-1,
                zmax=1,
                title="Correlation Matrix of Numerical Attributes",
                aspect="auto"
            )
            apply_blue_white_chart(fig_corr, height=400, show_legend=False)
            st.plotly_chart(fig_corr, use_container_width=True, config={'responsive': True, 'displayModeBar': False})
        else:
            st.info("ℹ️ At least 2 numerical features are required to compute a correlation matrix.")

        # Data Table View & CSV Download
        st.markdown("### 📋 Filtered Cohort Data View")
        st.caption(f"Showing {len(filtered_df):,} records currently filtered.")
        st.dataframe(filtered_df.head(15), use_container_width=True)

        csv_filtered = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv_filtered,
            file_name=f"filtered_{filename if filename else 'student_data.csv'}",
            mime="text/csv"
        )


# =========================================================
# 4. PAGE: PREDICT STUDENT PERFORMANCE
# =========================================================
def page_predict():
    st.markdown("""
        <div class="dashboard-header-card">
            <h2>🎯 Individual Student Performance Prediction</h2>
            <p>Predict academic final grade (G3) using the genuine pre-trained Machine Learning pipeline model.</p>
        </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("""
            ⚠️ **Trained Model Pipeline Unavailable!**
            
            The required model artifact `Model/best_model.pkl` could not be located in this environment.
            
            **How to resolve:**
            1. Locally, run: `python src/train_model.py` to train and generate the model.
            2. For Streamlit Cloud: Ensure `Model/best_model.pkl` is committed and pushed to GitHub (verify it is not excluded in `.gitignore`).
            
            *All other modules (Dashboard, Upload Data, Exploratory Analysis) remain fully functional.*
        """)
        return
    if default_df is None:
        st.error("❌ The base training dataset `Dataset/student-mat.csv` is missing.")
        return

    # Extract required feature names
    expected_cols = default_df.drop(columns=['G1', 'G2', 'G3'], errors='ignore').columns.tolist()

    st.markdown("""
    > **Authentic ML Model Schema:**
    > Predictions are generated by the Scikit-Learn Pipeline (`GradientBoostingRegressor`) trained strictly on authentic UCI student features without target leakage from `G1` or `G2`.
    """)

    with st.form("single_prediction_form"):
        st.subheader("📝 Enter Student Attributes")

        academic_lifestyle_cols = ['studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet']
        family_personal_cols = ['school', 'sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'romantic']
        health_habits_cols = ['famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'traveltime']

        covered_cols = set(academic_lifestyle_cols + family_personal_cols + health_habits_cols)
        other_expected_cols = [c for c in expected_cols if c not in covered_cols]

        input_dict = {}

        def render_feature_input(col, container):
            series = default_df[col]
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
                    help=f"Valid range in dataset: {c_min} to {c_max}"
                )
            else:
                choices = sorted([str(x) for x in series.dropna().unique().tolist()])
                return container.selectbox(
                    f"`{col}`",
                    choices,
                    key=f"inp_{col}",
                    help=f"Valid options: {', '.join(choices)}"
                )

        # Section 1: Demographics & Family
        st.markdown("##### 👤 Demographic & Family Background")
        c1, c2, c3 = st.columns(3)
        col_list_1 = [col for col in family_personal_cols if col in expected_cols]
        for i, col in enumerate(col_list_1):
            col_container = [c1, c2, c3][i % 3]
            input_dict[col] = render_feature_input(col, col_container)

        # Section 2: Academics & Support
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

        if other_expected_cols:
            st.markdown("##### 🔍 Additional Attributes")
            co1, co2, co3 = st.columns(3)
            for i, col in enumerate(other_expected_cols):
                col_container = [co1, co2, co3][i % 3]
                input_dict[col] = render_feature_input(col, col_container)

        submit_btn = st.form_submit_button("🔮 Predict Final Grade")

    if submit_btn:
        missing_form_features = [col for col in expected_cols if col not in input_dict]
        if missing_form_features:
            st.error(f"❌ Missing required features: {', '.join(missing_form_features)}")
        else:
            try:
                input_df = pd.DataFrame([input_dict])[expected_cols]
                predicted_val = model.predict(input_df)[0]
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
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=clipped_val,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Predicted G3 Grade", 'font': {'size': 18, 'family': 'Manrope', 'color': '#173B67'}},
                        gauge={
                            'axis': {'range': [0, 20], 'tickwidth': 1, 'tickcolor': "#64748B"},
                            'bar': {'color': "#2563EB"},
                            'steps': [
                                {'range': [0, 9.9], 'color': "#FEE2E2"},
                                {'range': [10, 13.9], 'color': "#FEF3C7"},
                                {'range': [14, 20], 'color': "#DCFCE7"}
                            ],
                            'threshold': {
                                'line': {'color': "#DC2626", 'width': 4},
                                'thickness': 0.75,
                                'value': 9.9
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        height=270,
                        margin=dict(l=20, r=20, t=55, b=20),
                        paper_bgcolor="#FFFFFF",
                        plot_bgcolor="#FFFFFF",
                        font=dict(family="Inter, sans-serif", color="#172B4D")
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True, config={'responsive': True, 'displayModeBar': False})

            except Exception as ex:
                st.error(f"❌ Prediction failed: {str(ex)}")


# =========================================================
# 5. PAGE: BATCH PREDICTION
# =========================================================
def page_batch_prediction():
    st.markdown("""
        <div class="dashboard-header-card">
            <h2>📑 Batch CSV Prediction</h2>
            <p>Predict academic grades for multiple students simultaneously via CSV upload with instant validation.</p>
        </div>
    """, unsafe_allow_html=True)

    if model is None:
        st.error("""
            ⚠️ **Trained Model Pipeline Unavailable!**
            The required model artifact `Model/best_model.pkl` could not be located.
            Please ensure `Model/best_model.pkl` is trained and pushed to your GitHub repository.
            *(Note: Dataset upload and exploratory analysis remain active and functional).*
        """)
        return
    if default_df is None:
        st.error("❌ Reference dataset `Dataset/student-mat.csv` missing.")
        return

    expected_cols = default_df.drop(columns=['G1', 'G2', 'G3'], errors='ignore').columns.tolist()

    st.markdown(f"""
    ### 📋 Required Batch Schema
    The uploaded CSV must contain the following **{len(expected_cols)}** features:
    `{', '.join(expected_cols)}`
    
    *(Note: If the uploaded file contains columns `G1`, `G2`, or `G3`, they are safely ignored during prediction to prevent target contamination).*
    """)

    sample_batch_df = default_df.drop(columns=['G1', 'G2', 'G3'], errors='ignore').head(5)
    sample_csv = sample_batch_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Sample Batch Template (CSV)",
        data=sample_csv,
        file_name="sample_batch_template.csv",
        mime="text/csv",
        help="Download pre-formatted template with 5 real student sample rows."
    )

    st.write("")
    batch_file = st.file_uploader(
        "Upload Batch CSV file",
        type=['csv'],
        help="Upload CSV with student attributes matching the required feature set."
    )

    if batch_file is not None:
        try:
            raw_bytes = batch_file.getvalue()
            enc = 'utf-8'
            try:
                sample_str = raw_bytes[:4096].decode('utf-8')
            except UnicodeDecodeError:
                enc = 'latin-1'
                sample_str = raw_bytes[:4096].decode('latin-1')

            first_line = sample_str.split('\n')[0] if '\n' in sample_str else sample_str
            delimiter = ';' if ';' in first_line else (',' if ',' in first_line else '\t')
            batch_file.seek(0)
            batch_df = pd.read_csv(batch_file, sep=delimiter, encoding=enc)

            if batch_df.empty or len(batch_df) == 0:
                st.error("❌ The uploaded batch CSV contains NO records.")
            else:
                missing_features = [col for col in expected_cols if col not in batch_df.columns]
                if missing_features:
                    st.error(f"❌ Incompatible format! Missing **{len(missing_features)}** required columns: {', '.join(missing_features)}")
                else:
                    results_df = batch_df.copy()
                    
                    # Row-level validation: check for missing required values or non-convertible numbers
                    num_features = default_df[expected_cols].select_dtypes(include=np.number).columns.tolist()
                    
                    valid_mask = pd.Series(True, index=batch_df.index)
                    # Check nulls in required columns
                    has_nulls = batch_df[expected_cols].isnull().any(axis=1)
                    valid_mask = valid_mask & (~has_nulls)
                    
                    # Verify numeric coercibility
                    for nc in num_features:
                        coerced = pd.to_numeric(batch_df[nc], errors='coerce')
                        valid_mask = valid_mask & (~coerced.isna())

                    valid_count = int(valid_mask.sum())
                    invalid_count = len(batch_df) - valid_count

                    # Initialize predicted_final_grade column
                    results_df['predicted_final_grade'] = np.nan

                    if valid_count > 0:
                        X_valid = batch_df.loc[valid_mask, expected_cols].copy()
                        for nc in num_features:
                            X_valid[nc] = pd.to_numeric(X_valid[nc], errors='coerce')
                        preds = model.predict(X_valid)
                        preds_clipped = np.clip(preds, 0.0, 20.0).round(2)
                        results_df.loc[valid_mask, 'predicted_final_grade'] = preds_clipped

                    if invalid_count > 0:
                        results_df['validation_status'] = "Valid"
                        results_df.loc[~valid_mask, 'validation_status'] = "Incompatible / Missing attributes"
                        results_df.loc[~valid_mask, 'predicted_final_grade'] = "Validation Error"
                        st.warning(f"⚠️ Validation Notice: **{valid_count}** of **{len(batch_df)}** records were compatible and predicted. **{invalid_count}** records had missing or invalid values and were preserved without prediction.")
                    else:
                        st.success(f"✅ Schema & data validated successfully! All **{len(batch_df)}** student records predicted.")

                    # Summary Metrics
                    st.markdown("### 📊 Batch Prediction Results")
                    r_c1, r_c2, r_c3 = st.columns(3)
                    r_c1.metric("Total Submitted Students", len(results_df))
                    
                    valid_preds_num = pd.to_numeric(results_df['predicted_final_grade'], errors='coerce').dropna()
                    if not valid_preds_num.empty:
                        r_c2.metric("Mean Predicted Grade", f"{valid_preds_num.mean():.2f} / 20")
                        r_c3.metric("Students At-Risk (< 10)", int((valid_preds_num < 10).sum()))
                    else:
                        r_c2.metric("Mean Predicted Grade", "Not available")
                        r_c3.metric("Students At-Risk (< 10)", 0)

                    st.dataframe(results_df.head(15), use_container_width=True)
                    with st.expander("🔍 View All Batch Prediction Records", expanded=False):
                        st.dataframe(results_df, use_container_width=True)

                    result_csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions Result as CSV",
                        data=result_csv,
                        file_name="student_performance_predictions.csv",
                        mime="text/csv",
                        help="Export full original records with added predicted_final_grade column."
                    )

                    if not valid_preds_num.empty:
                        fig_batch_hist = px.histogram(
                            x=valid_preds_num,
                            nbins=15,
                            title="Distribution of Batch Predicted Final Grades",
                            color_discrete_sequence=['#2563EB'],
                            labels={'x': 'Predicted Final Grade'}
                        )
                        fig_batch_hist.update_layout(bargap=0.08)
                        apply_blue_white_chart(fig_batch_hist, height=340, show_legend=False)
                        st.plotly_chart(fig_batch_hist, use_container_width=True, config={'responsive': True, 'displayModeBar': False})

        except Exception as e:
            st.error(f"❌ Error processing batch file: {str(e)}")


# =========================================================
# 6. PAGE: MODEL PERFORMANCE
# =========================================================
def page_model_performance():
    st.markdown("""
        <div class="dashboard-header-card">
            <h2>📈 Machine Learning Model Performance Dashboard</h2>
            <p>Benchmark comparison of trained models evaluated on the held-out test dataset (20% split).</p>
        </div>
    """, unsafe_allow_html=True)

    candidate_results = [
        ROOT_DIR / "Output" / "model_results.csv",
        APP_DIR / "Output" / "model_results.csv",
        Path("Output/model_results.csv").resolve()
    ]
    found_path = None
    for p in candidate_results:
        if p.is_file():
            found_path = str(p)
            break

    if found_path is not None:
        metrics_df = pd.read_csv(found_path)

        st.markdown("""
        > **Evaluation Methodology:**
        > Metrics were computed strictly on the **unseen 20% test dataset** (`random_state=42`) using Scikit-Learn `Pipeline` with Standard Scaling and One-Hot Encoding. No data leakage from `G1` or `G2` was permitted.
        """)

        st.markdown("### 🏆 Model Comparison Benchmark")
        st.dataframe(
            metrics_df.style.highlight_max(subset=['R2 Score'], color='#DCFCE7')
                           .highlight_min(subset=['RMSE', 'MAE', 'MSE'], color='#DCFCE7'),
            use_container_width=True
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_r2 = px.bar(
                metrics_df,
                x='Model',
                y='R2 Score',
                color='Model',
                text_auto='.4f',
                title="R² Score Comparison (Higher is Better)",
                color_discrete_sequence=['#173B67', '#2563EB', '#38BDF8']
            )
            apply_blue_white_chart(fig_r2, height=320, show_legend=False)
            st.plotly_chart(fig_r2, use_container_width=True, config={'responsive': True, 'displayModeBar': False})

        with c2:
            fig_rmse = px.bar(
                metrics_df,
                x='Model',
                y='RMSE',
                color='Model',
                text_auto='.4f',
                title="Root Mean Squared Error (RMSE) (Lower is Better)",
                color_discrete_sequence=['#D97706', '#F59E0B', '#FBBF24']
            )
            apply_blue_white_chart(fig_rmse, height=320, show_legend=False)
            st.plotly_chart(fig_rmse, use_container_width=True, config={'responsive': True, 'displayModeBar': False})

        c3, c4 = st.columns(2)
        with c3:
            fig_mae = px.bar(
                metrics_df,
                x='Model',
                y='MAE',
                color='Model',
                text_auto='.4f',
                title="Mean Absolute Error (MAE)",
                color_discrete_sequence=['#2563EB', '#38BDF8', '#60A5FA']
            )
            apply_blue_white_chart(fig_mae, height=320, show_legend=False)
            st.plotly_chart(fig_mae, use_container_width=True, config={'responsive': True, 'displayModeBar': False})

        with c4:
            fig_mse = px.bar(
                metrics_df,
                x='Model',
                y='MSE',
                color='Model',
                text_auto='.4f',
                title="Mean Squared Error (MSE)",
                color_discrete_sequence=['#4F46E5', '#6366F1', '#818CF8']
            )
            apply_blue_white_chart(fig_mse, height=320, show_legend=False)
            st.plotly_chart(fig_mse, use_container_width=True, config={'responsive': True, 'displayModeBar': False})

        # Live Evaluation Verification
        with st.expander("⚡ Live Test-Set Evaluation (Verify Authentic Calculation)", expanded=False):
            st.write("Click below to re-evaluate all 3 models directly against the authentic held-out 20% test split from `Dataset/student-mat.csv` using Scikit-Learn live in Python.")
            if st.button("Run Live Verification on Test Split"):
                try:
                    import sys
                    src_dir = str(ROOT_DIR / 'src')
                    if src_dir not in sys.path:
                        sys.path.append(src_dir)
                    from preprocessing import load_data, clean_data, prepare_data
                    from evaluate_model import evaluate_predictions
                    from sklearn.linear_model import LinearRegression
                    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                    from sklearn.pipeline import Pipeline

                    eval_csv = ROOT_DIR / 'Dataset' / 'student-mat.csv'
                    if not eval_csv.is_file():
                        eval_csv = Path('Dataset/student-mat.csv').resolve()
                    raw_eval_df = load_data(str(eval_csv))
                    clean_eval_df = clean_data(raw_eval_df)
                    X_tr, X_te, y_tr, y_te, prep = prepare_data(clean_eval_df)

                    eval_models = {
                        'Linear Regression': LinearRegression(),
                        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
                        'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
                    }

                    live_res = []
                    for m_name, est in eval_models.items():
                        pipe = Pipeline([('preprocessor', prep), ('model', est)])
                        pipe.fit(X_tr, y_tr)
                        preds = pipe.predict(X_te)
                        live_res.append(evaluate_predictions(y_te, preds, m_name))

                    live_df = pd.DataFrame(live_res)
                    st.success(f"✅ Live evaluation completed on {len(y_te)} unseen test students!")
                    st.dataframe(live_df, use_container_width=True)
                except Exception as eval_err:
                    st.error(f"Live evaluation error: {str(eval_err)}")

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
        st.error("❌ Evaluation metrics file not found (`Output/model_results.csv`).")


# =========================================================
# Native Streamlit Top Navigation Router
# =========================================================
pages = [
    st.Page(page_home_dashboard, title="Home", icon=":material/home:", default=True),
    st.Page(page_upload_data, title="Upload Data", icon=":material/upload:"),
    st.Page(page_data_analysis, title="Data Analysis", icon=":material/analytics:"),
    st.Page(page_predict, title="Predict", icon=":material/psychology:"),
    st.Page(page_batch_prediction, title="Batch Prediction", icon=":material/table_chart:"),
    st.Page(page_model_performance, title="Model Performance", icon=":material/monitoring:"),
]

# Configure native top navigation
pg = st.navigation(pages, position="top")
pg.run()
