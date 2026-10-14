import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

numerical_data = np.random.randint(20, 80, size=95).tolist()
outliers = [150, 160, 170, 5, 10]
numerical_data.extend(outliers)

categories = ['A', 'B', 'C', 'D']
categorical_data = np.random.choice(categories, size=100)

df = pd.DataFrame({
    'numerical_column': numerical_data,
    'categorical_column': categorical_data
})

print(df.head())

df.to_csv('Sample_Data.csv', index=False)

print(df.shape)
df.info()
print(df.nunique())

print(df.isnull().sum())
print(df.duplicated().sum())

num_col = df.select_dtypes(include=[np.number]).columns
cat_col = df.select_dtypes(include=['object']).columns

print(f"numerical_data {num_col}")
print(f"categorical_data {cat_col}")

print(df.describe().T)

mean_value = df[num_col].mean()
median_value = df[num_col].median()
mode_value = df[num_col].mode()
std_dev = df[num_col].std()
variance = df[num_col].var()
range_value = df[num_col].max() - df[num_col].min()

print(f"Mean: {mean_value}")
print(f"Median: {median_value}")
print(f"Mode: {mode_value}")
print(f"Standard Deviation: {std_dev}")
print(f"Variance: {variance}")
print(f"Range: {range_value}")

plt.figure(figsize=(8,5))
sns.histplot(df[num_col], bins=30, kde=True)
plt.title('Histogram')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(df[num_col])
plt.title('Box Plot')
plt.show()

Q1 = df['numerical_column'].quantile(0.25)
Q3 = df['numerical_column'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['numerical_column'] < lower_bound) |
              (df['numerical_column'] > upper_bound)]['numerical_column']

print("Outliers:\n", outliers)

category_counts = df[cat_col].value_counts()
print(category_counts)

plt.figure(figsize=(8,5))
category_counts.plot(kind='bar')
plt.title('Bar Chart of Categories')
plt.show()

plt.figure(figsize=(8,5))
category_counts.plot(kind='pie', autopct='%1.1f%%')
plt.title('Pie Chart of Categories')
plt.ylabel('')
plt.show()