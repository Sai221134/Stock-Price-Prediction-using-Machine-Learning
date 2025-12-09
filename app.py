import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved model
with open('tesla_stock_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Streamlit app
st.set_page_config(page_title="Tesla Stock Movement Predictor", layout="centered")
st.title("📈 Tesla Stock Movement Predictor")
st.write("Predict whether the Tesla stock will go **up** or **down** the next day.")

# User inputs: now individual values
open_price = st.number_input("💰 Open Price", value=600.00, format="%.2f")
close_price = st.number_input("💵 Close Price", value=590.00, format="%.2f")
high_price = st.number_input("📈 High Price", value=620.00, format="%.2f")
low_price = st.number_input("📉 Low Price", value=580.00, format="%.2f")
is_quarter_end = st.selectbox("🗓️ Is it quarter end?", ["No", "Yes"])

# Feature engineering
open_close = open_price - close_price
low_high = low_price - high_price
quarter_flag = 1 if is_quarter_end == "Yes" else 0

# Combine input
user_input = np.array([[open_close, low_high, quarter_flag]])

# Scaling using the same scaler used in training (⚠️ ideally, load real saved scaler)
scaler = StandardScaler()
user_input_scaled = scaler.fit_transform(user_input)

# Predict
if st.button("Predict"):
    prediction = model.predict(user_input_scaled)[0]
    probability = model.predict_proba(user_input_scaled)[0][1]

    if prediction == 1:
        st.success(f"📈 The model predicts the stock **will go up** tomorrow (confidence: {probability*100:.2f}%)")
    else:
        st.error(f"📉 The model predicts the stock **will go down** tomorrow (confidence: {(1-probability)*100:.2f}%)")

st.markdown("---")
st.markdown("💡 _Note: This is a demo prediction tool using past stock movement patterns. Always do your own research before investing._")
