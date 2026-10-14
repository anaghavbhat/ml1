import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")

iris.head()

iris.shape

print("\nBasic Information about Dataset:")
print(iris.info())

print("\nSummary Statistics:")
print(iris.describe())

print("\nMissing Values in Each Column:")
print(iris.isnull().sum())

iris.duplicated().sum()

plt.figure(figsize=(12, 8))
iris.hist(figsize=(12, 8), bins=30, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=iris)
plt.xticks(rotation=45)
plt.title("Boxplots of Features to Identify Outliers")
plt.show()

num_col = iris.select_dtypes(include=[np.number]).columns
cat_col = iris.select_dtypes(include=['object']).columns

print(f"numerical_data {num_col}")
print(f"categorical_data {cat_col}")

plt.figure(figsize=(10, 6))
corr_matrix = iris[num_col].corr('pearson')
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

label_encoder = LabelEncoder()
iris["species"] = label_encoder.fit_transform(iris["species"])

X = iris.drop(columns=["species"])
y = iris["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print('\nClassification Report:\n', classification_report(y_test, y_pred))
print('\nConfusion Matrix:\n', confusion_matrix(y_test, y_pred))