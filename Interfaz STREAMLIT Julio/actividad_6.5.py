import streamlit as st
import pandas as pd
import math
import numpy as np 
import matplotlib.pyplot as plt

#calcular distancia
@st.cache_data
def distance(x, y, z):
    return math.sqrt(x**2 + y**2 + z**2)

#leemos los datos y calculamos la distancia, multiplicamos por 3 ya que ese es el tiempo, esto depende de la sesión
# este código lo vimos con el profe juan manual 
def load_and_process_data(file):
    df = pd.read_csv(file)
    df["time"] = df.index * 3
    if {'Des_x', 'Des_y', 'Des_z'}.issubset(df.columns):
        df['Distancia'] = df.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
    else:
        st.error(f"❌ Las columnas necesarias para calcular la distancia no están presentes en {file}.")
        st.stop()
    return df


st.title("Equipo Los Ciclos For 🌀 ")
st.header("Actividad 6.5") 

df_p7s1 = load_and_process_data('P07_Sesion1.csv')
df_p7s2 = load_and_process_data('P07_Sesion2.csv')
df_p7s3 = load_and_process_data('P07_Sesion3.csv')
df_p7s4 = load_and_process_data('P07_Sesion4.csv')

df_p8s1 = load_and_process_data('P08_Sesion1.csv')
df_p8s2 = load_and_process_data('P08_Sesion2.csv')
df_p8s3 = load_and_process_data('P08_Sesion3.csv')
df_p8s4 = load_and_process_data('P08_Sesion4.csv')

df_p11s1 = load_and_process_data('P11_Sesion1.csv')
df_p11s2 = load_and_process_data('P11_Sesion2.csv')
df_p11s3 = load_and_process_data('P11_Sesion3.csv')
df_p11s4 = load_and_process_data('P11_Sesion4.csv')


paciente7 = [df_p7s1, df_p7s2, df_p7s3, df_p7s4]
paciente8 = [df_p8s1, df_p8s2, df_p8s3, df_p8s4]
paciente11 = [df_p11s1, df_p11s2, df_p11s3, df_p11s4]

sesiones = ['Sesión 1 🟡', 'Sesión 2 🔵', 'Sesión 3 🟢', 'Sesión 4 🟠']


st.sidebar.title("🧑‍⚕️ **Seguimiento de Pacientes**")
st.sidebar.subheader("Seleccione un paciente:")
var = st.sidebar.selectbox("Pacientes", ['Paciente 7', 'Paciente 8', 'Paciente 11'])

#titulo 
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Distancia Recorrida por Pacientes 🚶‍♂️</h1>", unsafe_allow_html=True)

#Función para graficar distancias, este código lo vimos con el profe juan manuel 
def graficar_distancias(paciente, titulo):
    valores = [sum(df['Distancia']) for df in paciente]
    fig, ax = plt.subplots()
    ax.bar(sesiones, valores, color=['#FFC107', '#03A9F4', '#8BC34A', '#FF5722'])
    ax.set_title(titulo, fontsize=16, color="#4CAF50")
    ax.set_xlabel("Sesiones", fontsize=12)
    ax.set_ylabel("Distancia (m)", fontsize=12)
    st.pyplot(fig)

#Mostrar gráfica según el paciente seleccionado
if var == "Paciente 7":
    graficar_distancias(paciente7, "📊 Distancia Recorrida - Paciente 7")
elif var == "Paciente 8":
    graficar_distancias(paciente8, "📊 Distancia Recorrida - Paciente 8")
elif var == "Paciente 11":
    graficar_distancias(paciente11, "📊 Distancia Recorrida - Paciente 11")

#Botones para sesiones con íconos
col1, col2, col3, col4 = st.columns(4)

#este código lo vimos con el profe juan manuel en donde usamos la columna time para graficar las emociones a lo largo del tiempo para cada sesión 
def mostrar_grafica_emociones(data, titulo):
    st.markdown(f"<h2 style='color: #FF9800;'>{titulo}</h2>", unsafe_allow_html=True)
    sentiment_columns = ['01_C', '02_A', '03_D', '04_M']
    fig, ax = plt.subplots(figsize=(10, 3))

    for i, col in enumerate(sentiment_columns):
        intervals = []
        start = None
        for j in range(len(data)):
            if data[col].iloc[j] == 1:
                if start is None:
                    start = data['time'].iloc[j]
            else:
                if start is not None:
                    end = data['time'].iloc[j - 1]
                    intervals.append((start, end))
                    start = None
        if start is not None:
            intervals.append((start, data['time'].iloc[-1]))

        for (start, end) in intervals:
            ax.hlines(y=i, xmin=start, xmax=end, color=plt.get_cmap("Set1")(i), linewidth=6)

    ax.set_yticks(range(len(sentiment_columns)))
    ax.set_yticklabels(sentiment_columns)
    ax.set_xlabel("Tiempo", fontsize=12)
    ax.set_title("Presencia de sentimientos en intervalos de tiempo", fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig)

#Botones para cada sesión
if col1.button("🟡 **Sesión 1**"):
    if var == "Paciente 7":
        mostrar_grafica_emociones(df_p7s1, "Emociones - Sesión 1")
    elif var == "Paciente 8":
        mostrar_grafica_emociones(df_p8s1, "Emociones - Sesión 1")
    elif var == "Paciente 11":
        mostrar_grafica_emociones(df_p11s1, "Emociones - Sesión 1")

if col2.button("🔵 **Sesión 2**"):
    if var == "Paciente 7":
        mostrar_grafica_emociones(df_p7s2, "Emociones - Sesión 2")
    elif var == "Paciente 8":
        mostrar_grafica_emociones(df_p8s2, "Emociones - Sesión 2")
    elif var == "Paciente 11":
        mostrar_grafica_emociones(df_p11s2, "Emociones - Sesión 2")

if col3.button("🟢 **Sesión 3**"):
    if var == "Paciente 7":
        mostrar_grafica_emociones(df_p7s3, "Emociones - Sesión 3")
    elif var == "Paciente 8":
        mostrar_grafica_emociones(df_p8s3, "Emociones - Sesión 3")
    elif var == "Paciente 11":
        mostrar_grafica_emociones(df_p11s3, "Emociones - Sesión 3")

if col4.button("🟠 **Sesión 4**"):
    if var == "Paciente 7":
        mostrar_grafica_emociones(df_p7s4, "Emociones - Sesión 4")
    elif var == "Paciente 8":
        mostrar_grafica_emociones(df_p8s4, "Emociones - Sesión 4")
    elif var == "Paciente 11":
        mostrar_grafica_emociones(df_p11s4, "Emociones - Sesión 4")





