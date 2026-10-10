import pandas as pd
import numpy as np

# Load dataset (Portuguese students - usually larger and better for regression)
# Or Math students (student-mat.csv)
df_mat = pd.read_csv('Dataset/student-mat.csv', sep=';')

print("=== Math Dataset Info ===")
print("Shape:", df_mat.shape)
print("\nMissing Values:\n", df_mat.isnull().sum()[df_mat.isnull().sum() > 0])
print("\nColumns:\n", df_mat.columns.tolist())
print("\nData Types:\n", df_mat.dtypes)

print("\nDuplicate Rows:", df_mat.duplicated().sum())

print("\nHead:\n", df_mat.head())
print("\nTarget Variable 'G3' Stats:\n", df_mat['G3'].describe())
