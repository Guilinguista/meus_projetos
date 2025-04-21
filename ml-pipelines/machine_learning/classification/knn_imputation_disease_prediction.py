# Disease Prediction with KNN Imputation and Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score
)

# Load dataset
df = pd.read_csv("frmgham2.csv")

# Select relevant features
columns = [
    "SEX", "TOTCHOL", "AGE", "SYSBP", "DIABP", "CURSMOKE",
    "CIGPDAY", "BMI", "DIABETES", "BPMEDS", "HEARTRTE",
    "GLUCOSE", "educ", "HDLC", "LDLC", "ANYCHD", "PERIOD"
]
df = df[columns]

# Filter to most recent period
df = df[df["PERIOD"] == 3].drop(columns=["PERIOD"])

# Visualize missing data
msno.matrix(df)
plt.title("Missing Data Matrix")
plt.show()

# Separate features and target
X = df.drop(columns=["ANYCHD"])
y = df["ANYCHD"]

# Convert to numpy
X = X.to_numpy()
y = y.to_numpy()

# Imputation using KNN
print("Missing before imputation:", np.isnan(X).sum())
imputer = KNNImputer()
X_imputed = imputer.fit_transform(X)
print("Missing after imputation:", np.isnan(X_imputed).sum())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.3, random_state=42, stratify=y
)

# Train a classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# AUC Score
auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC Score: {auc:.4f}")

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
