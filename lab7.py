import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv("BostonHousing.csv")

df = data.copy()

df['CRIM'] = df['CRIM'].fillna(df['CRIM'].mean())
df['ZN'] = df['ZN'].fillna(df['ZN'].mean())
df['CHAS'] = df['CHAS'].fillna(df['CHAS'].mode()[0])
df['INDUS'] = df['INDUS'].fillna(df['INDUS'].mean())
df['AGE'] = df['AGE'].fillna(df['AGE'].median())
df['LSTAT'] = df['LSTAT'].fillna(df['LSTAT'].median())

df['CHAS'] = df['CHAS'].astype('int')

for i in df.columns:
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    df[i].hist(bins=20, alpha=0.5, color='b', edgecolor='black')
    plt.title(f'Histogram of {i}')
    plt.xlabel(i)
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.boxplot(df[i], vert=False)
    plt.title(f'Boxplot of {i}')
    plt.show()

corr = df.corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.xticks(rotation=90, ha='right')
plt.yticks(rotation=0)
plt.title("Correlation Matrix Heatmap")
plt.show()

X = df.drop('MEDV', axis=1)
y = df['MEDV']

scale = StandardScaler()
X_scaled = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

data = sns.load_dataset('mpg')

df = data.copy()
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

numerical = df.select_dtypes(include=['int', 'float']).columns
categorical = df.select_dtypes(include=['object']).columns

print(numerical)
print(categorical)

for i in numerical:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    df[i].hist(bins=20, alpha=0.5, color='b', edgecolor='black')
    plt.title(f'Histogram of {i}')
    plt.xlabel(i)
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.boxplot(df[i], vert=False)
    plt.title(f'Boxplot of {i}')
    plt.show()

for col in categorical:
    plt.figure(figsize=(6, 6))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index[:10])
    plt.title(f'Countplot of {col}')
    plt.xticks(rotation=90)
    plt.show()

X = df[['horsepower']]
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared: {r2}')

X_plot = np.linspace(X['horsepower'].min(), X['horsepower'].max(), 100).reshape(-1, 1)
X_plot_poly = poly.transform(X_plot)
y_plot = model.predict(X_plot_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue')
plt.plot(X_plot, y_plot, color='red')
plt.title("Polynomial Regression")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.show()