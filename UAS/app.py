import pandas as pd
import numpy as np
import streamlit as st
import pickle
import matplotlib.pyplot as plt

# Load models
fish_model_svm = pickle.load(open('UAS/Supervised/SVM/SVM_fish.pkl', 'rb'))
fruit_model_svm = pickle.load(open('Supervised/SVM/SVM_fruit.pkl', 'rb'))
pumpkin_model_svm = pickle.load(open('Supervised/SVM/SVM_pumpkin.pkl', 'rb'))
fish_model_pcp = pickle.load(open('Supervised/Perceptron/PCP_fish.pkl', 'rb'))
fruit_model_pcp = pickle.load(open('Supervised/Perceptron/PCP_fruit.pkl', 'rb'))
pumpkin_model_pcp = pickle.load(open('Supervised/Perceptron/PCP_pumpkin.pkl', 'rb'))
fish_model_lr = pickle.load(open('Supervised/Logistic Regression/LR_fish.pkl', 'rb'))
fruit_model_lr = pickle.load(open('Supervised/Logistic Regression/LR_fruit.pkl', 'rb'))
pumpkin_model_lr = pickle.load(open('Supervised/Logistic Regression/LR_pumpkin.pkl', 'rb'))

# Set page title and configuration
st.set_page_config(page_title="Prediction with ML Models", layout="wide")
st.title("🌱 **Machine Learning Prediction** Using SVM, Perceptron, Logistic Regression")

# Input section for category and algorithm selection
st.markdown("### Select Classification Category and Algorithm")
option = st.selectbox("Choose Classification Category", ("Fish", "Fruit", "Pumpkin"))

# Hide algorithm selector if 'Wine' is chosen
if option != "Wine":
    algorithm = st.selectbox("Choose Algorithm", ("SVM", "Perceptron", "Logistic Regression"))
else:
    algorithm = None  # For wine, algorithm is not required

st.markdown("---")

# Dictionaries for fish, fruit, and pumpkin types
fish_types = {
    0: "Anabas testudineus",
    1: "Coilia dussumieri",
    2: "Otolithoides biauritus",
    3: "Otolithoides pama",
    4: "Pethia conchonius",
    5: "Polynemus paradiseus",
    6: "Puntius lateristriga",
    7: "Setipinna taty",
    8: "Sillaginopsis panijus"
}

fruit_types = {0: "Grapefruit", 1: "Orange"}

pumpkin_types = {0: "Çerçevelik", 1: "Ürgüp Sivrisi"}

# Input form based on category
st.markdown("---")
with st.form(key='prediction_form'):
    if option == "Fish":
        st.markdown("<h3 style='color: #2F4F4F;'>🦈 Input Fish Data</h3>", unsafe_allow_html=True)
        weight = st.number_input('Fish Weight (gram)', min_value=0.0, format="%.2f")
        length = st.number_input('Fish Length (cm)', min_value=0.0, format="%.2f")
        height = st.number_input('Fish Height (cm)', min_value=0.0, format="%.2f")

        submit = st.form_submit_button(label='Predict')

        if submit:
            input_data = np.array([weight, length, height]).reshape(1, -1)

            if algorithm == "SVM":
                prediction = fish_model_svm.predict(input_data)
            elif algorithm == "Perceptron":
                prediction = fish_model_pcp.predict(input_data)
            else:  # Logistic Regression
                prediction = fish_model_lr.predict(input_data)

            fish_result = fish_types.get(prediction[0], "Unknown")
            st.success(f"### Predicted Fish Type: {fish_result}")

    elif option == "Fruit":
        st.markdown("<h3 style='color: #FF6347;'>🍊 Input Fruit Data</h3>", unsafe_allow_html=True)
        diameter = st.number_input('Fruit Diameter (cm)', min_value=0.0, format="%.2f")
        weight = st.number_input('Fruit Weight (gram)', min_value=0.0, format="%.2f")
        red = st.number_input('Red Color Score', 0, 255, 0)
        green = st.number_input('Green Color Score', 0, 255, 0)
        blue = st.number_input('Blue Color Score', 0, 255, 0)

        submit = st.form_submit_button(label='Predict')

        if submit:
            input_data = np.array([diameter, weight, red, green, blue]).reshape(1, -1)

            if algorithm == "SVM":
                prediction = fruit_model_svm.predict(input_data)
            elif algorithm == "Perceptron":
                prediction = fruit_model_pcp.predict(input_data)
            else:  # Logistic Regression
                prediction = fruit_model_lr.predict(input_data)

            fruit_result = fruit_types.get(prediction[0], "Unknown")
            st.success(f"### Predicted Fruit Type: {fruit_result}")

    elif option == "Pumpkin":
        st.markdown("<h3 style='color: #FF8C00;'>🎃 Input Pumpkin Data</h3>", unsafe_allow_html=True)
        area = st.number_input('Area (cm²)', min_value=0.0, format="%.2f")
        perimeter = st.number_input('Perimeter (cm)', min_value=0.0, format="%.2f")
        major_axis_length = st.number_input('Major Axis Length (cm)', min_value=0.0, format="%.2f")
        minor_axis_length = st.number_input('Minor Axis Length (cm)', min_value=0.0, format="%.2f")
        convex_area = st.number_input('Convex Area (cm²)', min_value=0.0, format="%.2f")
        equiv_diameter = st.number_input('Equivalent Diameter (cm)', min_value=0.0, format="%.2f")
        eccentricity = st.number_input('Eccentricity', min_value=0.0, format="%.2f")
        solidity = st.number_input('Solidity', min_value=0.0, format="%.2f")
        extent = st.number_input('Extent', min_value=0.0, format="%.2f")
        roundness = st.number_input('Roundness', min_value=0.0, format="%.2f")
        aspect_ratio = st.number_input('Aspect Ratio', min_value=0.0, format="%.2f")
        compactness = st.number_input('Compactness', min_value=0.0, format="%.2f")

        submit = st.form_submit_button(label='Predict')

        if submit:
            input_data = np.array([area, perimeter, major_axis_length, minor_axis_length, convex_area, equiv_diameter, eccentricity, solidity, extent, roundness, aspect_ratio, compactness]).reshape(1, -1)

            if algorithm == "SVM":
                prediction = pumpkin_model_svm.predict(input_data)
            elif algorithm == "Perceptron":
                prediction = pumpkin_model_pcp.predict(input_data)
            else:  # Logistic Regression
                prediction = pumpkin_model_lr.predict(input_data)

            pumpkin_result = pumpkin_types.get(prediction[0], "Unknown")
            st.success(f"### Predicted Pumpkin Type: {pumpkin_result}")
