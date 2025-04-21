# Data Cleaning and Feature Engineering Example

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("exemplos_limpeza_dados.csv")

# Preview dataset
print("Initial data shape:", df.shape)
print(df.head())

# Show dataset info
print("\nDataset Info:")
df.info()

# Remove duplicate rows if any
df.drop_duplicates(inplace=True)
print("\nDuplicates removed. New shape:", df.shape)

# Create a second version of the dataset that drops rows with missing values (for comparison)
df_dropped = pd.read_csv("exemplos_limpeza_dados.csv")
df_dropped.dropna(inplace=True)
print("\nShape after dropping rows with missing values:", df_dropped.shape)

# Summary statistics
print("\nSummary statistics:")
print(df.describe())

# Fill missing values
df["Age"].fillna(df["Age"].mean(), inplace=True)
df["Salary"].fillna(df["Salary"].median(), inplace=True)
print("\nMissing values filled. Updated dataset:")
print(df.head())

# Compute average age of people living in Spain
mean_age_spain = df[df["Country"] == "Spain"]["Age"].mean()
print(f"\nAverage age of people from Spain: {mean_age_spain:.2f}")

# Example of label encoding or mapping categories
# If you have a column with ordinal values:
# Replace "1st", "2nd", "3th" with numeric equivalents
ordinal_mapping = {
    "1st": 1,
    "2nd": 2,
    "3th": 3
}

if "Rank" in df.columns:
    df["Rank_encoded"] = df["Rank"].map(ordinal_mapping)
    print("\nOrdinal encoding applied to 'Rank' column:")
    print(df[["Rank", "Rank_encoded"]].head())

# Label Encoding (example)
if "Country" in df.columns:
    le = LabelEncoder()
    df["Country_encoded"] = le.fit_transform(df["Country"])
    print("\nLabel encoding applied to 'Country' column:")
    print(df[["Country", "Country_encoded"]].head())

# One-hot encoding for nominal categories (alternative to label encoding)
df_encoded = pd.get_dummies(df, drop_first=True)
print("\nFinal dataset with one-hot encoding:")
print(df_encoded.head())
