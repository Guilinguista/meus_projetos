# PCA Visualization – Breast Cancer Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

# Load breast cancer dataset
data = load_breast_cancer()
features = data.data
labels = data.target

# Combine features and labels
df = pd.DataFrame(features, columns=data.feature_names)
df["label"] = labels
df["label_name"] = df["label"].apply(lambda x: "Benign" if x == 1 else "Malignant")

# Apply PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(features)
pca_df = pd.DataFrame(data=principal_components, columns=["PC1", "PC2"])
pca_df["Label"] = df["label_name"]

# Visualization
plt.figure(figsize=(10, 6))
colors = {"Benign": "skyblue", "Malignant": "salmon"}
for label in pca_df["Label"].unique():
    subset = pca_df[pca_df["Label"] == label]
    plt.scatter(subset["PC1"], subset["PC2"], label=label, alpha=0.6, c=colors[label])

plt.title("PCA – Breast Cancer Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
