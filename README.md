# Student Performance Prediction System

## 1. Project Overview
The **Student Performance Prediction System** is a Machine Learning capstone project designed to predict a student's final academic grade (G3) based on demographic, social, and school-related features. It includes a complete end-to-end machine learning pipeline, from data extraction and exploratory data analysis (EDA) to model training and deployment via a Streamlit web application.

## 2. Problem Statement
Academic performance is influenced by numerous factors beyond just studying. Identifying students at risk of performing poorly can help educators provide timely interventions. The problem is to accurately forecast final academic scores using only background information and study habits, without relying on intermediate grades.

## 3. Objectives
* Perform Exploratory Data Analysis to identify factors affecting student grades.
* Prevent data leakage by removing intermediate grades (`G1`, `G2`).
* Train and compare multiple Regression models to predict final grades (`G3`).
* Deploy the best-performing model using an interactive, user-friendly Streamlit web application.

## 4. Dataset Source
* **Repository:** UCI Machine Learning Repository
* **Dataset ID:** 320 (Student Performance)
* **Link:** [UCI Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)

## 5. Technologies Used
* **Programming Language:** Python 3.14+
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-learn
* **Data Visualization:** Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **Persistence:** Joblib

## 6. Project Structure
```text
Student-Performance-Prediction/
├── Dataset/                   # Contains the raw CSV dataset (student-mat.csv).
├── Notebook/                  # student_performance_analysis.ipynb (EDA).
├── Model/                     # best_model.pkl (Saved preprocessing & prediction pipeline).
├── Streamlit_App/             # app.py (The web application script).
├── Output/                    # Saved evaluation charts and model_results.csv.
├── src/                       # Source scripts (preprocessing.py, train_model.py, etc.).
├── requirements.txt           # Project dependencies.
├── .gitignore                 # Files to ignore for GitHub.
└── README.md                  # Project documentation.
```

## 7. Data Preprocessing
* **Missing Values & Duplicates:** The dataset was verified to have no missing values or duplicates.
* **Target Leakage Prevention:** `G1` and `G2` (first and second period grades) were removed to prevent artificial accuracy and target leakage.
* **Categorical Encoding:** Applied `OneHotEncoder` to all categorical columns (e.g., `school`, `sex`, `guardian`).
* **Numerical Scaling:** Applied `StandardScaler` to all numerical columns (e.g., `age`, `absences`).
* **Pipeline Integration:** All preprocessing steps were bundled into a Scikit-learn `Pipeline` alongside the model to ensure consistent transformations during prediction.

## 8. EDA Methodology
* **Univariate Analysis:** Analyzed the distribution of the target variable (`G3`). It follows a roughly normal distribution, with a cluster of zeros representing dropouts or no-shows.
* **Bivariate Analysis:** Used Boxplots to compare features like `studytime` against `G3`, revealing slight positive trends.
* **Correlation Analysis:** Generated a correlation heatmap for numerical variables to find potential multicollinearity.

## 9. Models Trained
1. **Linear Regression:** Baseline model.
2. **Random Forest Regressor:** Tree-based ensemble capable of capturing non-linear relationships.
3. **Gradient Boosting Regressor:** Sequential tree-based model.

## 10. Actual Evaluation Results
The models were evaluated on a 20% hold-out test set:

| Model | MAE | MSE | RMSE | R² Score |
| :--- | :--- | :--- | :--- | :--- |
| Linear Regression | 2.156 | 8.189 | 2.861 | 0.1601 |
| Random Forest | 2.047 | 7.966 | 2.822 | 0.1830 |
| **Gradient Boosting** | **2.112** | **7.912** | **2.812** | **0.1885** |

*Gradient Boosting Regressor was selected as the final model due to its lowest RMSE and highest R² Score.*

## 11. Installation Instructions
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd Student-Performance-Prediction
   ```
2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 12. Local Execution Instructions
1. Make sure you have trained the model (if `Model/best_model.pkl` is missing):
   ```bash
   python src/train_model.py
   ```
2. Start the Streamlit App:
   ```bash
   streamlit run Streamlit_App/app.py
   ```
3. Access the dashboard at `http://localhost:8501`.

## 13. Deployment Instructions (Streamlit Community Cloud)
To deploy this project online for free:
1. Push this entire project folder to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Click **"New app"**.
4. Select your repository, branch (`main`), and set the Main file path to `Streamlit_App/app.py`.
5. Click **"Deploy"**. Streamlit will automatically read the `requirements.txt` and launch your app.

## 14. Limitations and Future Scope
* **Limitations:** Predicting academic performance based purely on demographic and social factors (without past academic history) inherently yields lower R² scores. The model struggles with extreme outliers (e.g., students who score 0).
* **Future Scope:** Gathering more contextual data (e.g., student assignments, teacher feedback) could significantly improve model accuracy. We could also implement Deep Learning models if the dataset size grows.
