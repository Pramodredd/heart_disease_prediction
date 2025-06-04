import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

st.set_page_config(page_title="Heart Disease Predictor", layout="wide")

# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    model_path = "./random_forest_model.pkl"  # <-- Path to your local model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = joblib.load(model_path)
    return model

model = load_model()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load and preprocess data
df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Train-test split (same logic as used during model training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=323)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# -------------------- Home --------------------
st.title("\U0001F493 Heart Disease Risk Prediction")
st.markdown("""
Welcome to the Heart Disease Predictor App. This tool uses a trained Random Forest model to estimate the risk of heart disease based on various health parameters.
""")

# -------------------- Sidebar Input --------------------
st.sidebar.header("Input Features")

def user_input():
    age = st.sidebar.slider('Age', 10, 50, 90)  # Min changed from 29 to 10
    sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
    cp = st.sidebar.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    trestbps = st.sidebar.slider('Resting Blood Pressure (trestbps)', 90, 200, 120)
    chol = st.sidebar.slider('Serum Cholestoral (chol)', 50, 200, 600)  # Min changed from 120 to 50
    fbs = st.sidebar.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
    restecg = st.sidebar.selectbox('Resting ECG (restecg)', [0, 1, 2])
    thalach = st.sidebar.slider('Max Heart Rate Achieved (thalach)', 70, 150, 210)
    exang = st.sidebar.selectbox('Exercise Induced Angina (exang)', [0, 1])
    oldpeak = st.sidebar.slider('ST depression (oldpeak)', 0.0, 6.2, 1.0)
    slope = st.sidebar.selectbox('Slope of peak exercise ST segment', [0, 1, 2])
    ca = st.sidebar.selectbox('Number of major vessels (ca)', [0, 1, 2, 3])
    thal = st.sidebar.selectbox('Thalassemia (thal)', [0, 1, 2, 3])

    data = {
        'age': age,
        'sex': 1 if sex == 'Male' else 0,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# -------------------- Prediction --------------------
st.subheader("\U0001F52C Predict Heart Disease")

# if st.button("Predict"):
#     prediction = model.predict(input_df)[0]
#     st.success(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.markdown(
            f"<div style='background-color:#FFCCCC; padding:20px; border-radius:10px;'>"
            f"<h3 style='color:#990000;'>Prediction: Heart Disease</h3>"
            f"</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div style='background-color:#CCFFCC; padding:20px; border-radius:10px;'>"
            f"<h3 style='color:#006600;'>Prediction: No Heart Disease</h3>"
            f"</div>",
            unsafe_allow_html=True
        )
# -------------------- Visualizations --------------------
import plotly.express as px

st.subheader("\U0001F4CA Data Insights & Analysis")

with st.expander("Personalized Visualizations"):
    sample_data = pd.read_csv("heart.csv")

    # Age Distribution with user's age
    fig_age = px.histogram(sample_data, x="age", nbins=30, title="Age Distribution")
    fig_age.add_vline(x=input_df["age"][0], line_dash="dash", line_color="red")
    fig_age.update_layout(showlegend=False)
    st.plotly_chart(fig_age, use_container_width=True)

    # Cholesterol Distribution with user's chol
    fig_chol = px.histogram(sample_data, x="chol", nbins=30, title="Cholesterol Distribution", color_discrete_sequence=["orange"])
    fig_chol.add_vline(x=input_df["chol"][0], line_dash="dash", line_color="red")
    fig_chol.update_layout(showlegend=False)
    st.plotly_chart(fig_chol, use_container_width=True)

    # Plot user's data vs target using Plotly
with st.expander("User vs Population (Target Comparison)"):
    # Load sample data
    sample_data = pd.read_csv("heart.csv")

    # Create scatter plot of age vs cholesterol, colored by target
    fig_user_vs_target = px.scatter(
        sample_data,
        x="age",
        y="chol",
        color=sample_data["target"].map({0: "No Disease", 1: "Heart Disease"}),
        labels={"color": "Target"},
        title="Age vs Cholesterol Colored by Target",
        opacity=0.6
    )

    # Add user's data point
    fig_user_vs_target.add_scatter(
        x=[input_df["age"][0]],
        y=[input_df["chol"][0]],
        mode='markers+text',
        marker=dict(color='red', size=12, symbol='x'),
        name='Your Input',
        text=["You"],
        textposition="top center"
    )

    st.plotly_chart(fig_user_vs_target, use_container_width=True)




# -------------------- Footer --------------------
st.markdown("---")
st.markdown("Developed by Pramod | Powered by Random Forest Model")


