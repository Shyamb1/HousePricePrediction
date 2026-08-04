import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# STEP 1: Load Dataset
# ==========================================================

df = pd.read_csv("cleaned_train.csv")

print("First 5 Rows:")
print(df.head())

# ==========================================================
# STEP 2: One-Hot Encoding
# ==========================================================

df = pd.get_dummies(df, drop_first=True)

print("\nShape after Encoding:", df.shape)

# ==========================================================
# STEP 3: Features and Target
# ==========================================================

X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)

# ==========================================================
# STEP 4: Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Save processed dataset
processed_df = pd.concat([X, y], axis=1)
processed_df.to_csv("processed_train.csv", index=False)

print("Processed dataset saved!")

# ==========================================================
# STEP 5: Linear Regression
# ==========================================================

print("\n========== Linear Regression ==========")

lr_model = LinearRegression()

lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

comparison_lr = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": lr_predictions
})

print(comparison_lr.head())

lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_predictions))
lr_r2 = r2_score(y_test, lr_predictions)

print("\nLinear Regression Performance")

print("MAE :", lr_mae)
print("RMSE:", lr_rmse)
print("R2  :", lr_r2)

# ==========================================================
# STEP 6: Tuned Random Forest
# ==========================================================

print("\n========== Tuned Random Forest ==========")

rf_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

comparison_rf = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": rf_predictions
})

print(comparison_rf.head())

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_r2 = r2_score(y_test, rf_predictions)

print("\nTuned Random Forest Performance")

print("MAE :", rf_mae)
print("RMSE:", rf_rmse)
print("R2  :", rf_r2)

# ==========================================================
# STEP 7: Model Comparison
# ==========================================================

comparison = pd.DataFrame({
    "Metric": ["MAE", "RMSE", "R2 Score"],
    "Linear Regression": [lr_mae, lr_rmse, lr_r2],
    "Tuned Random Forest": [rf_mae, rf_rmse, rf_r2]
})

print("\n========== Model Comparison ==========")
print(comparison)

# ==========================================================
# STEP 8: Feature Importance
# ==========================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== Top 10 Important Features ==========")

print(importance_df.head(10))

# Graph

top10 = importance_df.head(10)

plt.figure(figsize=(10,6))

plt.bar(top10["Feature"], top10["Importance"])

plt.title("Top 10 Important Features")

plt.xlabel("Features")

plt.ylabel("Importance")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

# ==========================================================
# STEP 9: Save Model
# ==========================================================

joblib.dump(rf_model, "house_price_model.pkl")

print("\nModel saved successfully!")

# ==========================================================
# STEP 10: Load Model
# ==========================================================

loaded_model = joblib.load("house_price_model.pkl")

print("Model loaded successfully!")

loaded_predictions = loaded_model.predict(X_test)

comparison_loaded = pd.DataFrame({
    "Actual": y_test.values,
    "Prediction": loaded_predictions
})

print("\nFirst Five Predictions")

print(comparison_loaded.head())

print("\nPrediction Values")

print(loaded_predictions[:5])