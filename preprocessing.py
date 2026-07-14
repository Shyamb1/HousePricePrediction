import pandas as pd

df = pd.read_csv("train.csv")

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

print("Missing Values")
print(missing_values)

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = df.select_dtypes(include=["object"]).columns

for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("Remaining Missing Values:")
print(df.isnull().sum().sum())

df.to_csv("cleaned_train.csv", index=False)

print("Cleaned dataset saved successfully!")