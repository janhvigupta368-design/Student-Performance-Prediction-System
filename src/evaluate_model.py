import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_predictions(y_true, y_pred, model_name):
    """Calculates evaluation metrics and returns a dictionary."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'Model': model_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    }

def plot_results(results_df, output_dir='Output'):
    """Plot model comparison and save."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot R2 Scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='R2 Score', data=results_df)
    plt.title('Model Comparison - R2 Score (Higher is better)')
    plt.ylim(0, 1)
    plt.savefig(os.path.join(output_dir, 'model_comparison_r2.png'))
    plt.close()
    
    # Plot RMSE Scores
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='RMSE', data=results_df)
    plt.title('Model Comparison - RMSE (Lower is better)')
    plt.savefig(os.path.join(output_dir, 'model_comparison_rmse.png'))
    plt.close()

def plot_residuals(y_true, y_pred, model_name, output_dir='Output'):
    """Plot residuals for a model."""
    os.makedirs(output_dir, exist_ok=True)
    residuals = y_true - y_pred
    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.title(f'Residual Plot: {model_name}')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.savefig(os.path.join(output_dir, f'residuals_{model_name.replace(" ", "_")}.png'))
    plt.close()
