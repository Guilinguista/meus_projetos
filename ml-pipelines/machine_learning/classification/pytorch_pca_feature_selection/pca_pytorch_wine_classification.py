# Import standard Python libraries for data handling
import pandas as pd
import numpy as np

# Import scikit-learn tools for preprocessing, PCA, and train/test splitting
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

# Import PyTorch libraries for building and training the model
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------
# STEP 1: Load dataset from the UCI repository
# ----------------------------------------
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')  # Read CSV using semicolon separator

# ----------------------------------------
# STEP 2: Separate features (X) and target (y)
# ----------------------------------------
X = df.drop("quality", axis=1)   # Features: all columns except 'quality'
y = df["quality"]                # Target: wine quality score (3 to 8)

# Convert to binary classification: good (>=6) vs bad (<6)
y = (y >= 6).astype(int)         # Now y = 1 if quality >= 6, else 0

# ----------------------------------------
# STEP 3: Split into train and test sets
# ----------------------------------------
# test_size=0.2 → 20% of the data will be for testing, 80% for training
# random_state=42 → ensures reproducibility; every run gives the same split
# (Fun fact: 42 is a joke from the book "The Hitchhiker’s Guide to the Galaxy" where 42 is 'the answer to life, the universe, and everything.')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------
# STEP 4: Standardize features (important before PCA)
# ----------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # Fit scaler on training data and transform it
X_test_scaled = scaler.transform(X_test)         # Transform test data using the same scaler

# ----------------------------------------
# STEP 5: Apply PCA to reduce dimensionality
# ----------------------------------------
pca = PCA(n_components=5)                        # Keep only 5 principal components
X_train_pca = pca.fit_transform(X_train_scaled)  # Fit PCA on train and transform train data
X_test_pca = pca.transform(X_test_scaled)        # Transform test data with the same PCA

# Print how much variance each component explains
print("Explained variance ratio:", pca.explained_variance_ratio_)

# ----------------------------------------
# STEP 6: Convert NumPy arrays to PyTorch tensors
# ----------------------------------------
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)  
# .unsqueeze(1) turns shape from (N,) to (N, 1) to match model output shape

X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

# ----------------------------------------
# STEP 7: Define a simple feedforward neural network
# ----------------------------------------
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(5, 16)     # Input layer: 5 features → 16 neurons
        self.fc2 = nn.Linear(16, 1)     # Output layer: 16 → 1 output neuron
        self.relu = nn.ReLU()           # Activation between layers
        self.sigmoid = nn.Sigmoid()     # Output activation for binary classification

    def forward(self, x):
        x = self.relu(self.fc1(x))      # Apply first layer + ReLU
        x = self.sigmoid(self.fc2(x))   # Apply second layer + Sigmoid
        return x

# Instantiate the model
model = SimpleMLP()

# ----------------------------------------
# STEP 8: Define loss function and optimizer
# ----------------------------------------
criterion = nn.BCELoss()                          # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer with learning rate 0.01

# ----------------------------------------
# STEP 9: Train the model
# ----------------------------------------
epochs = 50                                       # Number of times to go through the dataset
for epoch in range(epochs):
    model.train()                                 # Set model to training mode
    optimizer.zero_grad()                         # Clear gradients from previous step

    output = model(X_train_tensor)                # Forward pass
    loss = criterion(output, y_train_tensor)      # Calculate loss
    loss.backward()                               # Backpropagation
    optimizer.step()                              # Update weights

    # Print training loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

# ----------------------------------------
# STEP 10: Evaluate model on test data
# ----------------------------------------
model.eval()                                      # Set model to evaluation mode
with torch.no_grad():                             # Disable gradient tracking
    preds = model(X_test_tensor)                  # Forward pass on test data
    preds_label = (preds >= 0.5).float()          # Convert probabilities to 0 or 1
    acc = (preds_label == y_test_tensor).float().mean()  # Accuracy calculation
    print(f"Test Accuracy: {acc.item():.4f}")
