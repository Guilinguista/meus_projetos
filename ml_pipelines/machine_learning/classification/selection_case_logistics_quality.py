# Selection Case – Product Quality and Damage Analysis

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset (replace 'data.csv' with your actual file name)
df = pd.read_csv("replacement_parts.csv")  # <-- you can rename your CSV accordingly

# Quick overview
print("Dataset preview:")
print(df.head())
print("\nColumn names:", df.columns)

# Clean column names if necessary
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Frequency analysis
quality_issues = df[df["problem_type"] == "quality"]
damage_issues = df[df["problem_type"] == "damage"]

# Most problematic items by type
most_quality_product = quality_issues["product"].value_counts().idxmax()
most_quality_count = quality_issues["product"].value_counts().max()

most_damage_product = damage_issues["product"].value_counts().idxmax()
most_damage_count = damage_issues["product"].value_counts().max()

print(f"\nProduct with most quality issues: {most_quality_product} ({most_quality_count} reports)")
print(f"Product with most damage issues: {most_damage_product} ({most_damage_count} reports)")

# Visualization
plt.figure(figsize=(12, 5))
sns.countplot(data=quality_issues, x="product", order=quality_issues["product"].value_counts().index)
plt.title("Products with Most Quality Complaints")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
sns.countplot(data=damage_issues, x="product", order=damage_issues["product"].value_counts().index)
plt.title("Products with Most Damage Issues")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Suggestion based on findings
print(f"\n🔧 Suggestion:")
print(f"To reduce damage in '{most_damage_product}', consider reinforcing packaging, especially around fragile components.")

# Email to supplier
email = f"""Subject: Quality Concerns Regarding '{most_quality_product}'

Dear Supplier,

After reviewing recent product replacements, we found that '{most_quality_product}' stands out as the item with the highest number of quality-related complaints ({most_quality_count} cases).

We kindly request your team to investigate potential causes in the production process and propose improvements to reduce these issues. Please let us know if you need additional data or support from our side.

Best regards,  
Guilherme Antônio Silva
"""

print("\n📧 Suggested Email:")
print(email)
