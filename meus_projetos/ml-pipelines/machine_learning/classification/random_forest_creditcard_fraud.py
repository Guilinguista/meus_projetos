# Credit Card Fraud Detection using Random Forest

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Problem overview:
# The dataset contains credit card transactions made in September 2013 by European cardholders.
# It includes 492 frauds out of 284,807 transactions (~0.17% are fraud), which makes it highly imbalanced.
# Features are PCA-transformed due to confidentiality (V1 to V28), and the target is 'Class' (1 = fraud, 0 = legit).

# Load dataset
url = "https://raw.githubusercontent.com/Guilinguista/assets/main/creditcard.csv"
df = pd.read_csv(url)

# Basic inspection
print("Dataset shape:", df.shape)
print(df["Class"].value_counts(normalize=True))

# Features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print(f"ROC AUC Score: {roc_auc:.4f}")

# Optional: visualize class distribution
sns.countplot(data=df, x="Class")
plt.title("Class Distribution (0 = Non-Fraud, 1 = Fraud)")
plt.tight_layout()
plt.show()
