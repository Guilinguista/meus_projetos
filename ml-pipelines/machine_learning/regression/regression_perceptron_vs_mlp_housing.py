# Regression of Housing Prices using Perceptron and MLP

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load housing data
url = "https://tinyurl.com/alugueis-sp-df"
df = pd.read_csv(url)

# Filter for properties that are for sale
df = df[df["Negotiation Type"] == "sale"].drop(columns=["Latitude", "Longitude"])

# Define features and target
X = df.drop(columns=["Price", "District", "Negotiation Type", "Property Type"])
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Perceptron Regressor (not ideal for regression, just for demonstration)
perceptron = Perceptron(max_iter=1000, tol=1e-3)
perceptron.fit(X_train_scaled, y_train)
y_pred_perceptron = perceptron.predict(X_test_scaled)

# MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)

# Evaluation function
def evaluate_regression(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "R²": r2}

# Collect results
results = [
    evaluate_regression(y_test, y_pred_perceptron, "Perceptron"),
    evaluate_regression(y_test, y_pred_mlp, "MLP Regressor")
]

results_df = pd.DataFrame(results)
print("\nEvaluation Results:")
print(results_df)

# Plot predicted vs true prices
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_mlp, alpha=0.5, label="MLP")
plt.scatter(y_test, y_pred_perceptron, alpha=0.5, label="Perceptron")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs True Prices")
plt.legend()
plt.tight_layout()
plt.show()
