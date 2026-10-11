import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from preprocessing import load_data, clean_data, prepare_data
from evaluate_model import evaluate_predictions, plot_results, plot_residuals

ROOT_DIR = Path(__file__).resolve().parent.parent

def main():
    # 1. Load and clean data
    filepath = ROOT_DIR / 'Dataset' / 'student-mat.csv'
    if not filepath.exists():
        filepath = Path('Dataset/student-mat.csv').resolve()
    if not filepath.exists():
        print(f"Error: Dataset not found at {filepath}")
        return
        
    df = load_data(str(filepath))
    df = clean_data(df)
    
    # 2. Prepare data (this drops G1, G2 to prevent target leakage and handles train-test split)
    print("Preparing data and preventing target leakage...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)
    
    # 3. Define Models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
    }
    
    results = []
    best_r2 = -float('inf')
    best_model_name = ""
    best_pipeline = None
    
    model_dir = ROOT_DIR / 'Model'
    output_dir = ROOT_DIR / 'Output'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # 4. Train and Evaluate each model
    for name, estimator in models.items():
        print(f"Training {name}...")
        
        # Combine preprocessing and model into a single Pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', estimator)
        ])
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Predict on Test set
        y_pred = pipeline.predict(X_test)
        
        # Evaluate
        metrics = evaluate_predictions(y_test, y_pred, name)
        results.append(metrics)
        
        print(f"{name} Results: R2={metrics['R2 Score']:.4f}, RMSE={metrics['RMSE']:.4f}")
        
        # Plot Residuals
        plot_residuals(y_test, y_pred, name)
        
        # Select best model based on R2 Score
        if metrics['R2 Score'] > best_r2:
            best_r2 = metrics['R2 Score']
            best_model_name = name
            best_pipeline = pipeline

    # 5. Save results to CSV
    results_df = pd.DataFrame(results)
    results_file = output_dir / 'model_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"\nModel comparison saved to {results_file}")
    print(results_df)
    
    # 6. Plot model comparison
    plot_results(results_df, output_dir=str(output_dir))
    
    # 7. Save best model
    if best_pipeline is not None:
        model_path = model_dir / 'best_model.pkl'
        joblib.dump(best_pipeline, model_path)
        print(f"\nBest Model is {best_model_name} with R2 = {best_r2:.4f}.")
        print(f"Pipeline saved successfully at {model_path}.")
        
        # Analyzing limitations via Random Forest or GB feature importances could be done here, 
        # but residuals plots will help with visual inspection.

if __name__ == '__main__':
    main()
