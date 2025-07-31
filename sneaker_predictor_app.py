import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

# Load trained model and label encoders
model = joblib.load('best_xgb_model.pkl')
categorical_cols = ['Brand', 'Product Type', 'Gender', 'Category', 'Country']
encoders = {col: joblib.load(f'{col}_encoder.pkl') for col in categorical_cols}

st.title("👟 Sneaker Sales Predictor")
st.write("Enter sneaker details below to predict total sales amount ($).")

# Get label classes for dropdowns
brand_options = encoders['Brand'].classes_
product_options = encoders['Product Type'].classes_
gender_options = encoders['Gender'].classes_
category_options = encoders['Category'].classes_
country_options = encoders['Country'].classes_

# Input form
brand = st.selectbox("Brand", brand_options)
product_type = st.selectbox("Product Type", product_options)
gender = st.selectbox("Gender", gender_options)
category = st.selectbox("Category", category_options)
country = st.selectbox("Country", country_options)
quantity = st.slider("Quantity", 1, 5, 3)
unit_price = st.number_input("Unit Price ($)", min_value=10.0, max_value=500.0, value=100.0)

# Prediction
if st.button("Predict Sales Amount"):
    # Raw input dictionary
    input_data = {
        'Brand': brand,
        'Product Type': product_type,
        'Gender': gender,
        'Category': category,
        'Country': country,
        'Quantity': quantity,
        'Unit Price ($)': unit_price
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in categorical_cols:
        input_df[col] = encoders[col].transform(input_df[col])
    # Apply one-hot encoding using the same categories as training
    input_df_encoded = pd.get_dummies(input_df)

# Align input_df_encoded to the training columns (add missing ones as 0)
    expected_columns = model.get_booster().feature_names
    for col in expected_columns:
     if col not in input_df_encoded:
        input_df_encoded[col] = 0

# Ensure column order matches
    input_df_encoded = input_df_encoded[expected_columns]


    # Predict
    prediction = model.predict(input_df_encoded)[0]
    st.success(f"🤑 Predicted Sales Amount: **${prediction:,.2f}**")
