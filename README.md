# ❤️ Heart Disease Prediction using Machine Learning

This Streamlit web app predicts the likelihood of heart disease based on various medical parameters. It uses a trained machine learning model to assist in preliminary health analysis and decision-making.

## 🚀 Demo

Check out the live app here:  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heartdiseaseprediction7.streamlit.app/)  

---

## 📌 Features

- User-friendly interface built with **Streamlit**
- Accepts patient health data inputs
- Predicts likelihood of heart disease
- Displays prediction results instantly
- Machine Learning model trained on historical health data

---

## 🧠 Model & Technology

- Python 3.x
- Scikit-learn
- Pandas, NumPy
- Streamlit for frontend
- Trained classification model : Random Forest Model

---

## 📂 Project Structure
<pre> 📁 heart-disease-prediction/ ├── app.py # Streamlit application script ├── model.pkl # Trained machine learning model ├── requirements.txt # List of Python dependencies └── README.md # Project documentation </pre>

## 🔧 Setup Instructions
Follow these steps to run the project locally on your machine:
### 1. Clone the repository
  git clone git@github.com:Pramodredd/heart_disease_prediction.git
  
  cd heart-disease-prediction
### 2. Create a Virtual Environment (optional but recommended)
  python -m venv venv
## Activate the environment:
On Windows:
  venv\Scripts\activate

On macOS/Linux:
  source venv/bin/activate

### 3. Install Required Python Packages
pip install -r requirements.txt

### 4. Run the Streamlit App
streamlit run app.py
