import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

st.set_page_config(page_title='Análisis de Variables vs Tiempos', page_icon="⏱️",layout="wide")
st.title("Comparación de Variables vs Tiempo")

@st.cache 

def load_data():
    file_path = "P11_sesión_11abr.csv"
    df = pd.read_csv(file_path)
    df["time"] = df.index*3
    column_to_move = 'time'
    columns = [column_to_move]+[col for col in df.columns if col != column_to_move]

    df = df[columns]
    return df 

data = load_data()

if st.checkbox("Mostrar Dataframe"):
    st.subheader("Primeras filas del dataset")
    st.dataframe(data.head())

#Selección de variables a graficar 
st.sidebar.header("Configuración de Gráficas")
variables = ["Presn", "VelPr", "Des_z", "AcePr"]
selected_vars = st.sidebar.multiselect("Selecciona variables para analizar:", variables, default=variables)

#Configuración de colores 

color_map = {
    "Presn": "purple",
    "VelPr": "red",
    "Des_z": "blue",
    "AcePr": "gray"
}

#Crear figura y subplots 

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20,10))
axs = axs.flatten()

for i, var in enumerate(selected_vars[:4]):
    data.plot(x="time", y=var, ax=axs[i], color=color_map[var])
    axs[i].xaxis.set_minor_locator(MultipleLocator(250))
    axs[i].set_title(f"Gráfica de {var}")
    axs[i].set_xlabel("Tiempo")
    axs[i].set_ylabel(var) 

st.pyplot(fig) 

