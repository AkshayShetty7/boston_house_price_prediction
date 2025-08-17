# app.py
import nbimporter
import streamlit as st
import numpy as np
from Boston_housing import predict_price

st.set_page_config(page_title="Boston Housing Price Predictor", layout="centered")

st.title("🏠 Boston Housing Price Prediction")
st.write("Enter the housing details to predict the price (in $1000s).")

# Boston dataset features
features_names = [
    "CRIM: Per capita crime rate",
    "ZN: Proportion of residential land zoned for lots",
    "INDUS: Proportion of non-retail business acres per town",
    "CHAS: Charles River dummy variable (1 if tract bounds river, 0 otherwise)",
    "NOX: Nitric oxide concentration (parts per 10 million)",
    "RM: Average number of rooms per dwelling",
    "AGE: Proportion of owner-occupied units built before 1940",
    "DIS: Weighted distance to employment centers",
    "RAD: Index of accessibility to radial highways",
    "TAX: Full-value property tax rate per $10,000",
    "PTRATIO: Pupil-teacher ratio",
    "B: 1000(Bk - 0.63)^2 where Bk is proportion of Black residents",
    "LSTAT: % lower status of the population"
]

# Create input fields for user
inputs = []
for feature in features_names:
    val = st.number_input(feature, value=0.0)
    inputs.append(val)

if st.button("Predict Price"):
    prediction = predict_price(inputs)
    st.success(f"🏡 Predicted House Price: ${prediction*1000:.2f}")
