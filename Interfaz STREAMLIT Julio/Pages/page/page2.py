import streamlit as st 
from main import create_sidebar 

#Llamar a la función para crear el sidebar 

selected_page = create_sidebar()

st.title("Predicciones")