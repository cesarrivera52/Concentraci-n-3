import streamlit as st
import numpy as np 
import pandas as pd 
import plotly.express as px 

st.title("Ejemplo de contenedores") 

#Crear un contenedor 

with st.container():
    st.subheader("Contenedor 1: Información básica")
    st.write("Este contenedor agrupa datos y gráficos básicos")
    st.metric("Valor actual", 42, delta="1.5")

data = {
    'Fecha': pd.date_range('2023-01-01','2023-01-10'),
    'Valor_A': np.random.randint(10,50,10),
    'Valor_B': np.random.randint(20,600,10),    
}

df = pd.DataFrame(data)

with st.container():
    st.subheader("Contenedor 2: Detalles adicionales")
    st.text("Este contenedor presenta información adicional")
    chart_type = st.selectbox('Seleccionar tipo de gráfico',['Línea','Barra'])

    if chart_type == 'Línea':
        fig = px.line(df,x='Fecha', y=['Valor_A','Valor_B'], title = "Grafico de líneas")
    elif chart_type =='Barra':
        fig = px.bar(df,x='Fecha', y=['Valor_A','Valor_B'], title = "Grafico de barras")

st.plotly_chart(fig) 



#######################
with st.container():
    st.write("Esto está dentro del contenedor")

    st.bar_chart(np.random.rand(50,3))

st.write("Esto esta fuera del contenedor")

#########################
st.sidebar.header("Configuración")
selected_option = st.sidebar.selectbox('Seleccionar opción', ['Opción 1', 'Opción 2', 'Opción 3'])
st.sidebar.write(f'Opción seleccionada: {selected_option}')

long_text = "Lorem ipsum. "* 1000

with st.container(height=300):
    st.markdown(long_text) 

##########################

row1 = st.columns(3)
row2 = st.columns(3) 

for col in row1 + row2: 
    tile = col.container(height=120)
    tile.title(":balloon:")




