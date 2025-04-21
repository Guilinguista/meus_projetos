# Linear Regression – Health Insurance Charges

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"
df = pd.read_csv(url)

# Quick overview
print("Dataset shape:", df.shape)
print(df.head())

# Check for nulls
print("\nMissing values:")
print(df.isnull().sum())

# Encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)

# Exploratory plot
plt.figure(figsize=(10, 5))
sns.scatterplot(data=df, x="age", y="charges", hue="smoker")
plt.title("Insurance Charges vs Age (colored by Smoking)")
plt.tight_layout()
plt.show()

# Define features and target
X = df_encoded.drop("charges", axis=1)
y = df_encoded["charges"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² score:", r2_score(y_test, y_pred))

# Coefficients
coefficients = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print("\nModel Coefficients:")
print(coefficients.sort_values(by="Coefficient", key=abs, ascending=False))
