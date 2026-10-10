import streamlit as st
import pandas as pd
import joblib
import os
import io

# ----------------- Configuration -----------------
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #ffffff; }
    h1, h2, h3 { color: #1e3a8a; }
    .stButton>button { background-color: #2563eb; color: white; border-radius: 5px; }
    </style>
""", unsafe_allow_html=True)

# ----------------- Load Data & Model -----------------
@st.cache_data
def load_data():
    dataset_path = 'Dataset/student-mat.csv'
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path, sep=';')
    return None

@st.cache_resource
def load_model():
    model_path = 'Model/best_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

df = load_data()
model = load_model()

# ----------------- Navigation -----------------
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", [
    "Home Dashboard", 
    "Student Prediction", 
    "Batch Prediction", 
    "Model Analytics"
])

# ----------------- 1. Home Dashboard -----------------
if menu == "Home Dashboard":
    st.title("🎓 Student Performance Prediction System")
    st.write("Welcome to the browser-based Student Performance Predictor. This platform uses Machine Learning to accurately forecast students' final academic grades (G3) using their background details and study habits.")
    
    if df is not None:
        st.subheader("Dataset Information")
        st.write(f"**Total Student Records:** {df.shape[0]}")
        st.write(f"**Total Features:** {df.shape[1]}")
        st.dataframe(df.head())
        
        st.subheader("Dataset Statistics & Charts")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("Output/G3_Distribution.png"):
                st.image("Output/G3_Distribution.png", caption="Actual Distribution of Final Grades")
        with col2:
            if os.path.exists("Output/Studytime_vs_G3.png"):
                st.image("Output/Studytime_vs_G3.png", caption="Study Time vs Grades Analysis")
    else:
        st.error("Dataset missing. Ensure `Dataset/student-mat.csv` exists.")

# ----------------- 2. Student Prediction -----------------
elif menu == "Student Prediction":
    st.title("📊 Individual Student Prediction")
    st.write("Enter actual student details to generate a prediction using the trained model.")
    
    if model is None or df is None:
        st.error("Model or Dataset is missing. Cannot perform predictions.")
    else:
        features = df.drop(columns=['G1', 'G2', 'G3'], errors='ignore')
        
        with st.form("prediction_form"):
            st.subheader("Student Details")
            input_data = {}
            cols = st.columns(3)
            
            for i, col in enumerate(features.columns):
                c = cols[i % 3]
                if features[col].dtype == 'object':
                    unique_vals = features[col].unique().tolist()
                    input_data[col] = c.selectbox(f"{col}", unique_vals)
                else:
                    min_val = int(features[col].min())
                    max_val = int(features[col].max())
                    default_val = int(features[col].median())
                    input_data[col] = c.number_input(f"{col}", min_value=min_val, max_value=max_val, value=default_val)
            
            submit_button = st.form_submit_button("Predict Final Grade")
        
        if submit_button:
            input_df = pd.DataFrame([input_data])
            try:
                prediction = model.predict(input_df)[0]
                st.success(f"### Predicted Final Grade: {prediction:.2f} / 20")
                
                if prediction >= 16:
                    st.info("Performance Estimate: **Excellent** 🌟")
                elif prediction >= 12:
                    st.info("Performance Estimate: **Good** 👍")
                elif prediction >= 10:
                    st.info("Performance Estimate: **Pass** ✅")
                else:
                    st.warning("Performance Estimate: **Fail** ⚠️")
                st.caption("Disclaimer: This is a machine learning estimate, not a guaranteed outcome.")
            except Exception as e:
                st.error(f"Prediction Error: {str(e)}")

# ----------------- 3. Batch Prediction -----------------
elif menu == "Batch Prediction":
    st.title("📁 Batch CSV Prediction")
    st.write("Upload a CSV file containing multiple student records to predict their grades all at once.")
    
    if model is None or df is None:
        st.error("Model or Dataset missing. Cannot perform predictions.")
    else:
        expected_columns = df.drop(columns=['G1', 'G2', 'G3'], errors='ignore').columns.tolist()
        
        # Download template
        st.write("Need a template? Download one below:")
        template_df = df.drop(columns=['G1', 'G2', 'G3'], errors='ignore').head(5)
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV Template", csv_template, "template.csv", "text/csv")
        
        uploaded_file = st.file_uploader("Upload Student Data (CSV)", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file, sep=';' if ';' in uploaded_file.getvalue().decode('utf-8').split('\n')[0] else ',')
                
                # Validate columns
                missing_cols = [col for col in expected_columns if col not in batch_df.columns]
                if missing_cols:
                    st.error(f"Invalid format! Missing columns: {', '.join(missing_cols)}")
                else:
                    st.success("CSV validated successfully! Generating predictions...")
                    
                    # Ensure only expected columns are passed (ignore extra columns like names/IDs if any)
                    X_batch = batch_df[expected_columns]
                    
                    # Predict
                    predictions = model.predict(X_batch)
                    batch_df['Predicted_G3'] = predictions
                    
                    st.write("### Prediction Results")
                    st.dataframe(batch_df)
                    
                    # Download results
                    result_csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=result_csv,
                        file_name='student_predictions_results.csv',
                        mime='text/csv'
                    )
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")

# ----------------- 4. Model Analytics -----------------
elif menu == "Model Analytics":
    st.title("📈 Model Evaluation & Analytics")
    st.write("Compare the actual performance of the three trained Regression models (Linear Regression, Random Forest, Gradient Boosting).")
    
    results_path = "Output/model_results.csv"
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        st.subheader("Actual Calculated Metrics")
        st.dataframe(results_df.style.highlight_max(subset=['R2 Score'], color='lightgreen').highlight_min(subset=['RMSE'], color='lightgreen'))
        
        st.subheader("Performance Comparison Charts")
        col1, col2 = st.columns(2)
        with col1:
            if os.path.exists("Output/model_comparison_r2.png"):
                st.image("Output/model_comparison_r2.png", caption="R2 Score Comparison (Higher is better)")
        with col2:
            if os.path.exists("Output/model_comparison_rmse.png"):
                st.image("Output/model_comparison_rmse.png", caption="RMSE Comparison (Lower is better)")
                
        st.subheader("Residual Plot (Best Model)")
        if os.path.exists("Output/residuals_Gradient_Boosting.png"):
            st.image("Output/residuals_Gradient_Boosting.png", caption="Gradient Boosting Residuals")
    else:
        st.error("Evaluation results not found! Ensure model training was completed successfully.")
