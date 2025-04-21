# Hyperparameter Tuning – Grid Search vs Randomized Search (Random Forest Regressor)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset (insurance)
df = pd.read_csv("http://tinyurl.com/42rvaw4p")

# Basic preprocessing
df["smoker"] = df["smoker"].apply(lambda x: 1 if x == "yes" else 0)
df["sex"] = df["sex"].apply(lambda x: 1 if x == "female" else 0)
df.drop("region", axis=1, inplace=True)

# Features and target
X = df.drop("charges", axis=1)
y = df["charges"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define parameter grid
param_grid = {
    "n_estimators": [100, 300, 500],
    "max_features": ["auto", "sqrt"],
    "max_depth": [10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Grid Search
print("Running GridSearchCV...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, scoring="neg_mean_squared_error")
grid_search.fit(X_train, y_train)

# Randomized Search
print("Running RandomizedSearchCV...")
random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=10, cv=3, n_jobs=-1, random_state=42, scoring="neg_mean_squared_error")
random_search.fit(X_train, y_train)

# Evaluate both models
def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"--- {name} ---")
    print(f"RMSE: {np.sqrt(mse):.2f}")
    print(f"R² score: {r2:.4f}\n")

evaluate_model(grid_search.best_estimator_, "GridSearchCV")
evaluate_model(random_search.best_estimator_, "RandomizedSearchCV")

# Compare best parameters
print("Best parameters from GridSearchCV:")
print(grid_search.best_params_)

print("\nBest parameters from RandomizedSearchCV:")
print(random_search.best_params_)
