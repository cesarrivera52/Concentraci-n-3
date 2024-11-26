import numpy as np 
import pandas as pd 
import streamlit as st 

st.write("Primer intento de utilizar datos para crear una tabla")
st.write(pd.DataFrame({
    'first column':[1,2,3,4],
    'second column':[10,20,30,40]
}))

st.write("Nuevo Dataframe de datos aleatorios")
dataframe = pd.DataFrame(
   np.random.randn(10,20),
   columns=('col %d' % i for i in range(20))) 

st.dataframe(dataframe.style.highlight_max(axis=0)) 

def convert_df(df):
    return df.to_csv().encode("utf-8")

csv = convert_df(dataframe) 

##Crear el botón de descarga 
st.download_button(
    label = (f"Descargar datos como CSV"),
    data = csv,
    file_name = (f"large_df.csv"),
    mime = "text/csv" 
)

st.write("Nuevo dataframe de datos estáticos")
dataframe = pd.DataFrame(
   np.random.randn(10,20),
   columns=('col %d' % i for i in range(20))) 
st.table(dataframe)
 
