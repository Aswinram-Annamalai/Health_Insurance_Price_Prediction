#This is Code used in Visual Studio Code for stramlit purpose.
#Above this Insurance_App file you seeing .ipynb file that i create using Google Colab.

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load dataset
DF = pd.read_csv(r"C:\Users\produ\Downloads\Medical_insurance.csv")

# One-Hot Encode categorical variables
encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded = encoder.fit_transform(DF[['sex', 'smoker', 'region']])
df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['sex', 'smoker', 'region']))

DF_final = pd.concat([DF.drop(['sex', 'smoker', 'region'], axis=1), df_encoded], axis=1)

# Split data
X = DF_final.drop("charges", axis=1)
Y = DF_final["charges"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train model
Model = LinearRegression()
Model.fit(X_train, Y_train)

# Streamlit App
st.title("💊 Medical Insurance Charges Prediction")
st.write("Predict health insurance charges based on patient details.")

# User Inputs
#age = st.slider("Age", 18, 100, 30)
age = st.number_input("Age", 18, 100, 30,1)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
children = st.number_input("Number of Children", 0, 5, 0)
sex = st.radio("Sex", ["male", "female"])
smoker = st.radio("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Convert user input into model input format
input_dict = {
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex_male": [1 if sex == "male" else 0],
    "smoker_yes": [1 if smoker == "yes" else 0],
    "region_northwest": [1 if region == "northwest" else 0],
    "region_southeast": [1 if region == "southeast" else 0],
    "region_southwest": [1 if region == "southwest" else 0]
}

input_df = pd.DataFrame(input_dict)

# Prediction
if st.button("Predict Charges"):
    prediction = Model.predict(input_df)
    st.success(f"💰 Estimated Insurance Charge: ${prediction[0]:.2f}")

