# Perceptron – Learning the AND Logic Gate with Accuracy Evaluation

import numpy as np
from sklearn.metrics import accuracy_score

# Input dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target values (AND logic)
y = np.array([0, 0, 0, 1])

# Initialize weights
weights = np.array([0.0, 0.0])
learning_rate = 0.1

# Step activation function
def activation_function(summation):
    return 1 if summation >= 1 else 0

# Predict function using current weights
def predict(sample):
    summation = sample.dot(weights)
    return activation_function(summation)

# Training loop
def train():
    global weights
    total_error = 1
    epoch = 0

    while total_error != 0:
        print(f"Epoch {epoch + 1}")
        total_error = 0

        for i in range(len(X)):
            sample = X[i]
            expected = y[i]
            output = predict(sample)
            error = expected - output
            weights += learning_rate * error * sample
            print(f"  Input: {sample}, Expected: {expected}, Output: {output}, Error: {error}")
            print(f"  Updated weights: {weights}")
            total_error += abs(error)

        epoch += 1
        print()

    print("Training completed.")
    print("Final weights:", weights)

    # Final evaluation
    predictions = np.array([predict(x) for x in X])
    acc = accuracy_score(y, predictions)
    print(f"Final accuracy: {acc * 100:.2f}%")

# Run training
train()
