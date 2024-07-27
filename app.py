import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# Load the machine learning model
pipe = pk.load(open('LinearRegressionModel.pkl', 'rb'))

st.header('Car Price Prediction ML Model')

# Load the car data
cars_data = pd.read_csv('Cleaned_Car_data.csv')

# User inputs for the prediction
company = st.selectbox('Select Car Brand', cars_data['company'].unique())
filtered_cars = cars_data[cars_data['name'].str.split().str[0] == company]
name = st.selectbox('Select Model', filtered_cars['name'].unique())

year = st.slider('Car Manufactured Year', 1995, 2019)
kms_driven = st.slider('No of KM driven', 0, 400000)
fuel_type = st.selectbox('Fuel Type', cars_data['fuel_type'].unique())

# Prediction button
if st.button('Predict Price'):
    # Prepare the data for prediction
    input_data = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                              data=np.array([name, company, year, kms_driven, fuel_type]).reshape(1, 5))
    
    # Make the prediction
    predicted_price = pipe.predict(input_data)[0]
    predicted_price = abs(predicted_price)

    # Display the prediction
    st.write(f"The estimated price of the car is: ₹{predicted_price:.2f}")
