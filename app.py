import streamlit as st
import joblib
import pandas as pd

st.title("🏠 House Price Prediction")

overall_qual = st.slider(
    "Overall Quality",
    1,
    10,
    5
)

gr_liv_area = st.number_input(
    "Ground Living Area (sq ft)",
    value=1500
)

garage_cars = st.slider(
    "Garage Cars",
    0,
    5,
    2
)

year_built = st.number_input(
    "Year Built",
    value=2000
)

st.write("Selected Values")

st.write("Overall Quality:", overall_qual)
st.write("Living Area:", gr_liv_area)
st.write("Garage Cars:", garage_cars)
st.write("Year Built:", year_built)

model = joblib.load("models/house_price_model.pkl")