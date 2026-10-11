import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_data(filepath):
    """Load the student performance dataset."""
    return pd.read_csv(filepath, sep=';')

def clean_data(df):
    """Perform data cleaning (drop duplicates, handle missing values if any)."""
    df = df.drop_duplicates()
    # The dataset typically has no missing values, but if it does:
    df = df.dropna()
    return df

def get_preprocessor(numeric_cols, categorical_cols):
    """Returns a scikit-learn preprocessing pipeline."""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    return preprocessor

def prepare_data(df, target_col='G3', drop_cols=['G1', 'G2']):
    """
    Prepare data for modeling.
    Drops potential data leak columns like G1 and G2.
    Returns X_train, X_test, y_train, y_test, preprocessor.
    """
    # Drop data leak features if present
    cols_to_drop = [c for c in drop_cols if c in df.columns]
    X = df.drop(columns=[target_col] + cols_to_drop)
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=['int64', 'float64', 'number']).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=['number']).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor = get_preprocessor(numeric_cols, categorical_cols)
    
    return X_train, X_test, y_train, y_test, preprocessor
