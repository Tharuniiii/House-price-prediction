import numpy as np
import pandas as pd
import streamlit as st 
import pickle

model = pickle.load(open(r"C:\Users\Tharuni\Desktop\NIT\Aug month\18th,19th-regression frontned backedn\house price prediction\model\multiple_linear_regression_model.pkl",'rb'))

st.title("🏠 House Price Prediction (Multiple Regression)")

bedrooms = st.number_input("Number of Bedrooms", min_value=0, max_value=20, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=0, max_value=20, step=1)
sqft_living = st.number_input("Living Area (sqft)", min_value=200, max_value=20000, step=50)

if st.button("Predict Price"):
    # Prepare input as 2D array
    input_data = np.array([[bedrooms, bathrooms, sqft_living]])
    
    # Predict
    prediction = model.predict(input_data)
    
    st.success(f"💰 Estimated House Price: ${prediction[0]:,.2f}")