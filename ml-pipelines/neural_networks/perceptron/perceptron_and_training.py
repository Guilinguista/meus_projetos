# Perceptron – Learning the AND logic gate

import numpy as np

# Define training data
# Input pairs and their expected output (AND logic)
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])  # Output for AND logic

# Initialize weights to zero
weights = np.array([0.0, 0.0])

# Learning rate
learning_rate = 0.1

# Activation function: step function
def activation_function(value):
    return 1 if value >= 1 else 0

# Function to calculate the output based on current weights
def predict(inputs):
    weighted_sum = np.dot(inputs, weights)
    return activation_function(weighted_sum)

# Training loop
def train():
    global weights
    error_total = 1
    epoch = 0

    while error_total != 0:
        print(f"Epoch {epoch + 1}")
        error_total = 0
        for i in range(len(X)):
            input_sample = X[i]
            expected = y[i]

            output = predict(input_sample)
            error = expected - output

            # Weight update rule
            weights = weights + learning_rate * error * input_sample

            print(f"  Input: {input_sample}, Expected: {expected}, Output: {output}, Error: {error}")
            print(f"  Updated Weights: {weights}")

            error_total += abs(error)

        epoch += 1
        print()

    print("Training complete.")
    print("Final Weights:", weights)

# Run the training process
train()
