# Simple Perceptron Model – University Selection Scenario

import pandas as pd

# Define the weighted sum function
def weighted_sum(inputs, weights):
    total = 0
    for i in range(len(inputs)):
        total += inputs[i] * weights[i]
    return total

# Define the step activation function
def step_function(u, bias=0):
    return 1 if u + bias >= 0 else 0

# Example: basic test with arbitrary input and weights
inputs = [1, 2, 3]
weights = [0.1, 0.2, 0.3]
bias = 0

u = weighted_sum(inputs, weights)
result = step_function(u, bias)

print(f"Weighted sum: {u:.2f}")
if result == 1:
    print("Neuron activated.")
else:
    print("Neuron not activated.")

# Example 2: University selection based on decision matrix
# Load Excel file with candidate scores
# Columns: ['recognized_by_gov', 'payment_flexibility', 'teaching_mode', 'job_support']
df = pd.read_excel("escolha_faculdade.xlsx", index_col="ID")
universities = df.T  # transpose to have each school as a row

print("\nAvailable university options:")
print(universities)

# Define weights based on decision preferences (customizable)
# e.g., give higher importance to job support and flexibility
weights = [0.3, 0.2, 0.2, 0.3]

# Decision function using perceptron logic
def evaluate_school(scores, weights, bias=0):
    total = weighted_sum(scores, weights)
    return step_function(total, bias)

# Evaluate all universities
print("\nEvaluation results:")
for uni_name, row in universities.iterrows():
    scores = list(row)
    decision = evaluate_school(scores, weights, bias=-0.5)
    print(f"{uni_name}: {'ACCEPTED ✅' if decision else 'REJECTED ❌'}")
