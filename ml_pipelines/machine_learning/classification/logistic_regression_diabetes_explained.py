# Logistic Regression – Predicting Diabetes (Pima Indians Dataset)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset directly from a URL
# Source: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
url = "https://raw.githubusercontent.com/Guilinguista/assets/main/pima_diabetes.csv"
df = pd.read_csv(url)

# Check the shape and preview the first rows
print("Dataset shape:", df.shape)
print(df.head())

# Separate the features (X) and the target variable (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
# max_iter is set higher to ensure convergence
model = LogisticRegression(max_iter=1000)

# Fit the model to the training data
model.fit(X_train, y_train)

# Predict the target values for the test data
y_pred = model.predict(X_test)

# Evaluate the model using accuracy, confusion matrix, and classification report
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("Confusion Matrix:")
print(metrics.confusion_matrix(y_test, y_pred))

print("Classification Report:")
print(metrics.classification_report(y_test, y_pred))

# Optional: visualize the confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
