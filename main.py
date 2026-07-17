import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Load Cleaned dataset
df = pd.read_csv("cleaned_train.csv")

print(df.head())

df = pd.get_dummies(df, drop_first=True)

print("Shape after Encoding:", df.shape)

X = df.drop("SalePrice", axis=1)

y = df["SalePrice"]
print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("Training Data:", X_train.shape)

print("Testing Data:", X_test.shape)

processed_df = pd.concat([X, y], axis=1)

processed_df.to_csv("processed_train.csv", index=False)

print("Processed dataset saved!")

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Predicted Prices:")

print(predictions[:10])

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions
})

print(comparison.head(10))

mae = mean_absolute_error(y_test, predictions)

print("Mean Absolute Error:", mae)

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("Root Mean Squared Error:", rmse)

r2 = r2_score(y_test, predictions)

print("R2 Score:", r2)

print("\n===== Model Evaluation =====")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)
