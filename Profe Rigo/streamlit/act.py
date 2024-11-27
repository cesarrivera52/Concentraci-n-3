import pandas as pd 
import streamlit as st 
import numpy as np 

st.title("Interfaz usando columnas")
st.write("Aquí veremos el uso de las columnas") 




col1, col2, col3 = st.columns(3,gap='large')

with col1:  
    st.title('aaa')
    data = pd.DataFrame(np.random.randn(10, 1), columns=['Valores'])    
    st.bar_chart(data)
with col2:
    st.write('INFOOOO')
with col3:
    first_with=st.text_input('Inserte nombre')
    second_with=st.number_input('Inserte edad',min_value=1,max_value=100,step=10)
    st.button('Insertar')
    
    
    