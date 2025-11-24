



import streamlit as st
import pandas as pd
import pickle
import os
st.set_page_config(page_title="Éxito Cliente", layout="wide")

# -------------------- Cargar modelo --------------------
with open("./models/final_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------- Título --------------------
image_path = os.path.abspath("app_streamlit/logo_bm2.png")

st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: center;
        margin-bottom: 20px;">
        <img src="file://{image_path}" 
             style="width: 70%; max-width: 500px; height: auto;">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Éxito de conseguir al cliente")

# -------------- Inputs del usuario --------------

job = st.selectbox("Job", ["admin.", "technician", "blue-collar", "other"])
marital = st.selectbox("Marital status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
campaign = st.number_input("Campaign", min_value=1, max_value=50, value=1)
pdays = st.number_input("Days since last contact", min_value=-1, max_value=999, value=-1)
previous = st.number_input("Previous contacts", min_value=0, max_value=50, value=0)
default = st.selectbox("Has default?", ["yes", "no"])
housing = st.selectbox("Has housing loan?", ["yes", "no"])
loan = st.selectbox("Has personal loan?", ["yes", "no"])
poutcome = st.selectbox("Outcome of previous campaign", ["failure","success","other","unknown"])
pdays_contacted = st.number_input("Pdays contacted", min_value=0, max_value=50, value=0)

# -------------------- Crear dataframe --------------------

input_data = pd.DataFrame({
    "job": [job],
    "marital": [marital],
    "education": [education],
    "age": [age],
    "campaign": [campaign],
    "pdays": [pdays],
    "previous": [previous],
    "default": [default],
    "housing": [housing],
    "loan": [loan],
    "poutcome": [poutcome],
    "pdays_contacted": [pdays_contacted]
})

# ----------- Predicción ------
if st.button("Predecir"):
    pred_proba = model.predict_proba(input_data)[0][1]  # probabilidad de clase positiva
    pred_class = model.predict(input_data)[0]
    
    st.write(f"Predicción: {pred_class}")
    st.write(f"Probabilidad: {pred_proba:.2f}")
