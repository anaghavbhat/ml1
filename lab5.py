import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

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

def evaluate_knn(k_values, weights='uniform'):
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k, weights=weights)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        results[k] = {'accuracy': accuracy, 'f1_score': f1}
    return results

k_values = [1, 3, 5]
regular_knn_results = evaluate_knn(k_values, weights='uniform')
weighted_knn_results = evaluate_knn(k_values, weights='distance')

results_df = pd.DataFrame.from_dict({
    'Regular k-NN': regular_knn_results,
    'Weighted k-NN': weighted_knn_results
}, orient='index').T

results_df