import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create Output dir
os.makedirs('Output', exist_ok=True)

print("Loading data...")
df = pd.read_csv('Dataset/student-mat.csv', sep=';')

print(f"Shape: {df.shape}")
print(f"Duplicates: {df.duplicated().sum()}")
print(f"Missing Values: {df.isnull().sum().sum()}")

# Target Variable
print(f"\nTarget Variable (G3) Describe:\n{df['G3'].describe()}")

# Plot Target Variable Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['G3'], bins=20, kde=True)
plt.title('Distribution of Final Grades (G3)')
plt.savefig('Output/G3_Distribution.png')
plt.close()

# Potential Data Leakage: G1 and G2 are highly correlated with G3, 
# and often in real world we don't have G1, G2 when predicting G3, but we might. 
# We'll drop G1 and G2 to make it realistic for predicting before the course.
# Or we keep them, but let's decide to drop them as they represent future info.
# We'll just output the correlation to show it.

numeric_cols = df.select_dtypes(include=np.number).columns
plt.figure(figsize=(12,8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.savefig('Output/Correlation_Heatmap.png')
plt.close()

# Bivariate: studytime vs G3
plt.figure(figsize=(8,5))
sns.boxplot(x='studytime', y='G3', data=df)
plt.title('Study Time vs Final Grade')
plt.savefig('Output/Studytime_vs_G3.png')
plt.close()

print("EDA done and charts saved to Output/")
