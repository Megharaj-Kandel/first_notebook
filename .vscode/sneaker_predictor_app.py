
import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

# Load trained model and label encoder
model = joblib.load('best_xgb_model.pkl')
brand_encoder = joblib.load('brand_label_encoder.pkl')

st.title("👟 Sneaker Sales Predictor")
st.write("Enter sneaker details below to predict total sales amount ($).")

# Input features
brand = st.selectbox("Brand", brand_encoder.classes_)
product_type = st.selectbox("Product Type", ['Sneakers', 'Hoodie', 'Joggers', 'Cap', 'T-shirt'])
gender = st.selectbox("Gender", ['Male', 'Female', 'Unisex'])
category = st.selectbox("Category", ['Streetwear', 'Limited Edition', 'Sportswear'])
country = st.selectbox("Country", ['USA', 'UK', 'Japan', 'Germany', 'India', 'Canada', 'Australia'])
quantity = st.slider("Quantity", 1, 5, 3)
unit_price = st.number_input("Unit Price ($)", min_value=10.0, max_value=500.0, value=100.0)

# Predict button
if st.button("Predict Sales Amount"):
    # Encode brand
    encoded_brand = brand_encoder.transform([brand])[0]

    # Construct input dataframe
    input_df = pd.DataFrame([{
        'Brand': encoded_brand,
        'Product Type': product_type,
        'Gender': gender,
        'Category': category,
        'Country': country,
        'Quantity': quantity,
        'Unit Price ($)': unit_price
    }])

    # You may need to encode other columns too, depending on your training
    # For now, we’ll assume only brand was encoded and others were handled automatically

    # Predict
    prediction = model.predict(input_df)[0]
    st.success(f"🤑 Predicted Sales Amount: **${prediction:,.2f}**")
