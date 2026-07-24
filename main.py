import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
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

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_predictions)
)

rf_r2 = r2_score(y_test, rf_predictions)
print("\n===== Random Forest Evaluation =====")

print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R2  :", rf_r2)

comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2 Score"],
    "Linear Regression": [mae, rmse, r2],
    "Random Forest": [rf_mae, rf_rmse, rf_r2]
})

print("\n===== Model Comparison =====")
print(comparison)

feature_importance = rf_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_importance
})
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)
print("\n===== Top 10 Important Features =====")

print(importance_df.head(10))

top10 = importance_df.head(10)

plt.figure(figsize=(10,6))

plt.bar(top10["Feature"], top10["Importance"])

plt.title("Top 10 Important Features")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()
rf_model.fit(X_train, y_train)

joblib.dump(rf_model, "house_price_model.pkl")

print("Model saved successfully!")
loaded_model = joblib.load("house_price_model.pkl")

print("Model loaded successfully!")
loaded_predictions = loaded_model.predict(X_test)
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Prediction": loaded_predictions
})

print(comparison.head(10))
print("\nFirst Five Predictions")
print(loaded_predictions[:5])