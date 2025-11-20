import pandas as pd
from pipelines.generator import generate_synthetic
from pipelines.filter import filter_synthetic
from config.settings import EMBEDDER_NAME

# -----------------------------------------------------
# 1. Load Seed Dataset
# -----------------------------------------------------
seed_df = pd.read_csv("data/seed.csv")
print("Seed Data:")
print(seed_df.head())

# -----------------------------------------------------
# 2. Generate Synthetic Data
# -----------------------------------------------------
print("\nGenerating synthetic data...")
synthetic_df = generate_synthetic(seed_df)
print("\nRaw Synthetic Samples:")
print(synthetic_df.head())

# -----------------------------------------------------
# 3. Filter Synthetic Data
# -----------------------------------------------------
print("\nFiltering synthetic data...")
filtered_df = filter_synthetic(seed_df, synthetic_df, EMBEDDER_NAME)

filtered_df.to_csv("data/synthetic.csv", index=False)

print("\nFiltered Synthetic Data Saved:")
print(filtered_df.head())
