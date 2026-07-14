import pandas as pd

# Load dataset
df = pd.read_csv("train.csv")

print("First 5 Rows:")
print(df.head())

print("\nRows and Columns:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()

print("\n========== Statistical Summary ==========")
print(df.describe())

print("\n========== Missing Values ==========")
print(df.isnull().sum())

print("\n========== Duplicate Rows ==========")
print(df.duplicated().sum())

print("\n========== SalePrice Summary ==========")
print(df["SalePrice"].describe())

print("\n========== Data Types ==========")
print(df.dtypes)

