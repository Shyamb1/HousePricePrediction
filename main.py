import pandas as pd

import matplotlib.pyplot as plt

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

plt.figure(figsize=(10,6))

plt.hist(df["SalePrice"], bins=30)

plt.title("Distribution of House Prices")
plt.xlabel("Sale Price")
plt.ylabel("Number of Houses")

plt.show()

plt.figure(figsize=(8,5))

plt.boxplot(df["SalePrice"])

plt.title("Box Plot of SalePrice")

plt.show()

plt.figure(figsize=(10,6))

plt.scatter(df["GrLivArea"], df["SalePrice"])

plt.xlabel("Ground Living Area")

plt.ylabel("Sale Price")

plt.title("Living Area vs Sale Price")

plt.show()