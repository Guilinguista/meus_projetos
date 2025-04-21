# Housing Price Prediction using Perceptron and MLP Regressor

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score
)

# Load dataset
url = "https://tinyurl.com/alugueis-sp-df"
df = pd.read_csv(url)

# Filter for properties that are for sale and drop unused columns
df = df[df["Negotiation Type"] == "sale"]
df = df.drop(columns=["Latitude", "Longitude"])

# Define features and target
X = df.drop(columns=["Price", "District", "Negotiation Type", "Property Type"])
y = df["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Perceptron (for demonstration)
perceptron = Perceptron(max_iter=1000)
perceptron.fit(X_train_scaled, y_train)
y_pred_perceptron = perceptron.predict(X_test_scaled)

# Train MLP Regressor
mlp = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
mlp.fit(X_train_scaled, y_train)
y_pred_mlp = mlp.predict(X_test_scaled)

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    return {
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred, squared=False),
        "R²": r2_score(y_true, y_pred)
    }

# Collect evaluation results
results = [
    evaluate_model(y_test, y_pred_perceptron, "Perceptron"),
    evaluate_model(y_test, y_pred_mlp, "MLP Regressor")
]

results_df = pd.DataFrame(results)
print("\nEvaluation Results:")
print(results_df)

# Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_perceptron, label="Perceptron", alpha=0.5)
plt.scatter(y_test, y_pred_mlp, label="MLP Regressor", alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Housing Prices")
plt.legend()
plt.tight_layout()
plt.show()
