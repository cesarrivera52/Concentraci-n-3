import numpy as np 
import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt 
import io 

st.set_page_config(page_title='Exploración Airbnb', layout='wide')

st.title("Exploración de datos de Airbnb") 

file_path = "listings.csv" 
df = pd.read_csv(file_path) 

if st.checkbox('Mostrar Dataframe'):
    st.subheader("Primeras filas del dataset")
    st.dataframe(df.head()) 

st.sidebar.title("Opciones de visualización")
columna = st.sidebar.selectbox("Selecciona la columna a graficar", df.columns)
st.write(f"Histograma de {columna}")
st.bar_chart(df[columna].value_counts()) 

cities = df['neighbourhood_cleansed'].unique()
selected_city = st.sidebar.selectbox("selecciona una ciudad", cities) 
filtered_data = df[df['neighbourhood_cleansed'] == selected_city] 

st.subheader(f"Datos para {selected_city}")
st.dataframe(filtered_data[['name','price','property_type','room_type','latitude','longitude']]) 

st.subheader("Distribución de precios por ciudad")
fig, ax = plt.subplots() 
df[df['neighbourhood_cleansed'] == selected_city]['price'].hist(bins=10, ax=ax) 
ax.set_xticks(ax.get_xticks()[::600]) 
ax.set_title(f"Distribución de Precios en {selected_city}") 
ax.set_xlabel("Rango de precios")
ax.set_ylabel("Frecuencia") 
st.pyplot(fig) 

#@st.cache_data
#Guarda la gráfica como un objeto BytesIO 
img = io.BytesIO()
plt.savefig(img,format='png')
img.seek(0)

##Crear el botón de descarga 
st.download_button(
    label = (f"Descargar gráfica de {selected_city}"),
    data = img,
    file_name = (f"Gráfica de {selected_city}.png"),
    mime = "image/png" 
)

sample_fraction = 0.02 
sampled_data = filtered_data.sample(frac=sample_fraction, random_state=1)
st.subheader("Mapa de propiedades")
st.map(sampled_data[['latitude','longitude']]) 


