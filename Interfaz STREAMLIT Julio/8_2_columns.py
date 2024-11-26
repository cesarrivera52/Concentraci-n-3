import pandas as pd 
import streamlit as st 
import numpy as np 

st.title("Interfaz usando columnas")
st.write("Aquí veremos el uso de las columnas") 




col1, col2, col3 = st.columns(3, gap='large') 

with col1:  
    st.subheader("Columna 1")
    data = pd.DataFrame(np.random.randn(10, 1), columns=['Valores'])    
    st.bar_chart(data) 

with col2:
    st.subheader("Columna 2")
    st.write("Hola, aquí se muestra información")
    st.write("Aquí hay más información ")

with col3:
    st.subheader("Columna 3")
    nombre = st.text_input('Escribe tu nombre')
    edad = st.number_input('Escribe tu edad', min_value =1, max_value=100)  
    if st.button("Enviar"):
        st.write(f"nombre: {nombre} y edad: {edad}")



