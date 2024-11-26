import numpy as np 
import pandas as pd 
import streamlit as st 

if st.checkbox('Mostrar Dataframe'):
    chart_data = pd.DataFrame(
        np.random.randn(20,3),
        columns=['a','b','c']) 
    chart_data 

st.button('Haz clic aquí') 
st.radio('Elige un método de contacto', ['E-mail', 'Facebook', 'Insta']) 
st.selectbox('Elige un método de envío', ['Standard (5-15 días)', 'Express (2-5 días)']) 
st.multiselect('Elige un planeta', ['Saturno', 'Marte', 'Tierra']) 
st.select_slider('Elige una nota', ['Mala','Buena', 'Excelente']) 
st.slider('Elige un número',0,50)  