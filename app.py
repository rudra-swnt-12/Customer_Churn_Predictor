import streamlit as st
import numpy as np
import pandas as pd

import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

import pickle

# Load the model
model = tf.keras.models.load_model('model.h5')

# Load the encoders & scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('OneHotEncoder_geography.pkl', 'rb') as file:
    onehot_encoder_geography = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app
st.title("Customer Churn Prediction")

# Input fields
geography = st.selectbox('Geography', onehot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One Hot Encoding for Geography
geography_encoded = onehot_encoder_geography.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(
    geography_encoded, 
    columns=onehot_encoder_geography.get_feature_names_out(['Geography'])
    )

# Combine one-hot encoded columns with input data
input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_scaled)
prediction_proba = prediction[0][0]

print(f" Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 0.5:
    st.write(f"Customer is likely to churn with a probability of {prediction_proba:.2f}")
else:
    st.write(f"Customer is not likely to churn with a probability of {1 - prediction_proba:.2f}")
