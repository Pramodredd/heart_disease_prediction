# ❤️ Heart Disease Prediction using Machine Learning

This Streamlit web app predicts the likelihood of heart disease based on various medical parameters. It uses a trained machine learning model to assist in preliminary health analysis and decision-making.

## 🚀 Demo

Check out the live app here:  
[![Open in Streamlit](https://heartdiseaseprediction7.streamlit.app/)  

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
heart-disease-prediction/
├── app.py
├── model.pkl
├── requirements.txt
└── README.md

## 🔧 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
