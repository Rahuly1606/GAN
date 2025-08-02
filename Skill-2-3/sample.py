from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load dataset
data = fetch_california_housing()
X = data.data
feature_names = data.feature_names

# Create DataFrame and save
df = pd.DataFrame(X, columns=feature_names)
df.head(20).to_csv("sample_input.csv", index=False)  # Save first 20 rows

