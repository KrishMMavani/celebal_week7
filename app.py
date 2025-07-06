# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load model
with open('model/iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# App title
st.title("🌸 Iris Flower Prediction App")
st.write("Input flower measurements to classify the species.")

# Sidebar Inputs
st.sidebar.header("Input Features")
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', 4.0, 8.0, 5.1)
    sepal_width  = st.sidebar.slider('Sepal width (cm)', 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider('Petal length (cm)', 1.0, 7.0, 1.4)
    petal_width  = st.sidebar.slider('Petal width (cm)', 0.1, 2.5, 0.2)
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    return pd.DataFrame([data])

df = user_input_features()

# Display input
st.subheader("User Input Parameters")
st.write(df)

# Make prediction
prediction = model.predict(df)[0]
prediction_proba = model.predict_proba(df)

# Show results
st.subheader("Prediction")
st.write(f"Predicted class: **{target_names[prediction]}**")

st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame(prediction_proba, columns=target_names)
st.write(proba_df)

# Plot
st.subheader("📊 Probability Distribution")
fig, ax = plt.subplots()
ax.bar(target_names, prediction_proba[0], color=['red', 'green', 'blue'])
ax.set_ylabel("Probability")
st.pyplot(fig)
