import streamlit as st
import pandas as pd
import joblib

# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠"
)

st.title("🏠 House Price Prediction")

st.write(
    "Predict house prices using a trained Random Forest model."
)

st.markdown("---")


# ==========================================================
# LOAD MODEL AND DATA
# ==========================================================

model = joblib.load(
    "models/house_price_model.pkl"
)

feature_columns = joblib.load(
    "models/feature_columns.pkl"
)

processed_data = pd.read_csv(
    "data/processed_train.csv"
)

# Remove target column
if "SalePrice" in processed_data.columns:
    processed_data = processed_data.drop(
        "SalePrice",
        axis=1
    )


# ==========================================================
# USER INPUTS
# ==========================================================

overall_qual = st.slider(
    "Overall Quality",
    min_value=1,
    max_value=10,
    value=5
)

gr_liv_area = st.number_input(
    "Ground Living Area (sq ft)",
    min_value=300,
    max_value=6000,
    value=1500
)

garage_cars = st.slider(
    "Garage Cars",
    min_value=0,
    max_value=5,
    value=2
)

year_built = st.number_input(
    "Year Built",
    min_value=1800,
    max_value=2026,
    value=2000
)

lot_area = st.number_input(
    "Lot Area (sq ft)",
    min_value=500,
    max_value=100000,
    value=8000
)

total_bsmt_sf = st.number_input(
    "Total Basement Area (sq ft)",
    min_value=0,
    max_value=5000,
    value=1000
)


# ==========================================================
# PREDICTION
# ==========================================================

st.markdown("---")

if st.button("🏠 Predict House Price"):

    # Take first processed house as a baseline
    input_data = processed_data.iloc[[0]].copy()

    # Make sure feature order matches training
    input_data = input_data.reindex(
        columns=feature_columns
    )

    # Replace values with user input
    input_data["OverallQual"] = overall_qual
    input_data["GrLivArea"] = gr_liv_area
    input_data["GarageCars"] = garage_cars
    input_data["YearBuilt"] = year_built
    input_data["LotArea"] = lot_area
    input_data["TotalBsmtSF"] = total_bsmt_sf

    # Make prediction
    prediction = model.predict(input_data)

    predicted_price = prediction[0]

    st.success(
        f"🏠 Estimated House Price: ${predicted_price:,.2f}"
    )