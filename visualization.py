import pandas as pd

import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("train.csv")

plt.figure(figsize=(10,6))

plt.hist(df["SalePrice"], bins=30)

plt.title("Distribution of House Prices")
plt.xlabel("Sale Price")
plt.ylabel("Number of Houses")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))

plt.boxplot(df["SalePrice"])

plt.title("Box Plot of SalePrice")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,6))

plt.scatter(df["GrLivArea"], df["SalePrice"],alpha = 0.6)

plt.xlabel("Ground Living Area")

plt.ylabel("Sale Price")

plt.title("Living Area vs Sale Price")
plt.grid(True)
plt.show()