import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

df1 = sns.load_dataset('titanic')
df1

print("\nBasic Information about Dataset:")
print(df1.info())

print("\nSummary Statistics:")
print(df1.describe())

print("\nMissing Values in Each Column:")
print(df1.isnull().sum())

df1.duplicated().sum()

columns_remove = ['deck', 'embarked', 'alive', 'pclass']
df = df1.drop(columns=columns_remove)

df['age'].fillna(round(df['age'].mean()), inplace=True)
df['embark_town'].mode()[0]
df['embark_town'].fillna(df['embark_town'].mode()[0], inplace=True)

df.isnull().sum()

df.drop_duplicates(inplace=True)

df.duplicated().sum()

plt.figure(figsize=(12, 8))
df.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df)
plt.xticks(rotation=45)
plt.title("Boxplots of Features to Identify Outliers")
plt.show()

num_col = df.select_dtypes(include=[np.number]).columns.tolist()
cat_col = df.select_dtypes(include=['object']).columns
print(f"numerical_data: {num_col}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['age'], y=df['fare'])
plt.title("Scatter Plot: Age vs Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

sns.pairplot(df, diag_kind='kde')
plt.show()

correlation = df[['age', 'fare']].corr('pearson')
print("Pearson Correlation Coefficient:\n", correlation)

covariance = df[['age', 'fare']].cov()
print("Covariance Matrix:\n", covariance)

plt.figure(figsize=(10, 6))
corr_matrix = df[num_col].corr('pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()