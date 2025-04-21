# Simple Perceptron Implementation – Manual + Practical Application

# Example 1 – Manual Perceptron Calculation

# Inputs
X = [1, 2, 3]  # Input values
W = [0.1, 0.2, 0.3]  # Weights
b = 0  # Bias term

# Weighted sum function
def weighted_sum(X, W, bias):
    total = 0
    for i in range(len(X)):
        total += X[i] * W[i]
    return total + bias

# Step activation function
def step_function(u):
    return 1 if u >= 0 else 0

# Apply functions
u = weighted_sum(X, W, b)
output = step_function(u)

print(f"Weighted sum result: {u:.2f}")
print("Neuron activated!" if output == 1 else "Neuron not activated.")


# Example 2 – Using Perceptron for Decision-Making Based on Excel Input

import pandas as pd

# Load decision dataset (e.g., university selection scores)
# Replace with actual file path if needed
data = pd.read_excel("escolha_faculdade.xlsx", index_col="ID")

# Transpose for viewing schools in rows
schools = data.T
print("\nSchool evaluation data:")
print(schools)

# Optional: apply decision rule based on a threshold
# For example: consider a school 'approved' if the weighted sum of its scores exceeds a threshold
def decide_school_acceptance(scores, weights, bias=0):
    result = weighted_sum(scores, weights, bias)
    return step_function(result)

# Example usage with first row
sample_school = list(schools.iloc[0])
sample_weights = [0.3] * len(sample_school)  # You can customize these weights
decision = decide_school_acceptance(sample_school, sample_weights)

print(f"Decision for {schools.index[0]}:", "Accept" if decision else "Reject")
