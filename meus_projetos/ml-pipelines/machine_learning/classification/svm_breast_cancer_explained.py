# Support Vector Machine (SVM) – Breast Cancer Classification

# --------------------------------------------
# 📦 Step 1 – Import necessary libraries
# --------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# --------------------------------------------
# 🧬 Step 2 – Load the dataset
# --------------------------------------------
# This dataset contains measurements of breast tumors along with labels:
# 0 = Malignant (cancerous), 1 = Benign (non-cancerous)
data = load_breast_cancer()
features = data.data
labels = data.target

# --------------------------------------------
# 🧱 Step 3 – Create a DataFrame for easier handling
# --------------------------------------------
df = pd.DataFrame(features, columns=data.feature_names)
df["label"] = labels

# --------------------------------------------
# ✂️ Step 4 – Split features and target
# --------------------------------------------
X = df.drop("label", axis=1)  # Features
y = df["label"]               # Target (class)

# --------------------------------------------
# 🔀 Step 5 – Split the dataset into training and testing sets
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# --------------------------------------------
# 📏 Step 6 – Standardize the feature data
# --------------------------------------------
# SVMs are sensitive to feature scales, so standardization is important
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------
# 🧠 Step 7 – Train the SVM classifier
# --------------------------------------------
# We'll use the RBF kernel, which is good for non-linear separation
svm_model = SVC(kernel='rbf')
svm_model.fit(X_train_scaled, y_train)

# --------------------------------------------
# 🔍 Step 8 – Make predictions on the test set
# --------------------------------------------
y_pred = svm_model.predict(X_test_scaled)

# --------------------------------------------
# 🧪 Step 9 – Evaluate the model
# --------------------------------------------
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------
# 🎯 Conclusion
# --------------------------------------------
# SVMs are powerful classifiers, especially effective in high-dimensional spaces.
# Standardization is key to achieving good performance.
