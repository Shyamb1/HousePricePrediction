import pandas as pd

# Load the dataset
df = pd.read_csv("train.csv")

print("First 5 Rows:")
print(df.head())

print("\nRows and Columns:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nDataset Information:")
df.info()
print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nSale Price Summary:")
print(df["SalePrice"].describe())
print(df.head(10))

print("\nDuplicate Rows:")
print(df.duplicated().sum())