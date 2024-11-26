import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib import ticker
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

st.set_page_config(page_title='Analisis de Datos del Socio Formador', layout='wide')

#Formula de Distancia ----------------------------------------------------------------------------------------------------
def distance(x, y, z):
    return math.sqrt(x**2 + y**2 + z**2)

#Funcion para cargar datos del archivo ----------------------------------------------------------------------------------------------------
def load_data(filename):
    #Cargar archivo
    file_path = filename
    df = pd.read_csv(file_path)

    #Generar columna de tiempo
    df['time'] = df.index*7
    column_to_move = 'time'
    columns = [column_to_move]+[col for col in df.columns if col != column_to_move]
    df = df[columns]

    #Generr columna de distancia
    df['Distancia'] = df.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)

    return df

#Carga de datos ----------------------------------------------------------------------------------------------------
#Paciente07
p07_session1 = load_data('data/paciente07/P07_2019_01ene02_v07_atribMOVPREFAE_etiq.csv')
p07_session2 = load_data('data/paciente07/P07_2018_12dic03_v07_atribMOVPREFAE_etiq.csv')
p07_session3 = load_data('data/paciente07/P07_2018_12dic10_v07_atribMOVPREFAE_etiq.csv')
p07_session4 = load_data('data/paciente07/P07_2018_12dic17_v07_atribMOVPREFAE_etiq.csv')

#Paciente08
p08_session1 = load_data('data/paciente08/P08_2019_01ene02_v07_atribMOVPREFAE_etiq.csv')
p08_session2 = load_data('data/paciente08/P08_2019_01ene09_v07_atribMOVPREFAE_etiq.csv')
p08_session3 = load_data('data/paciente08/P08_2019_01ene18_v07_atribMOVPREFAE_etiq.csv')
p08_session4 = load_data('data/paciente08/P08_2019_01ene21_v07_atribMOVPREFAE_etiq.csv')

#Paciente11
p11_session1 = load_data('data/paciente11/P11_2019_04abr11_v07_atribMOVPREFAE_etiq.csv')
p11_session2 = load_data('data/paciente11/P11_2019_04abr17_v07_atribMOVPREFAE_etiq.csv')
p11_session3 = load_data('data/paciente11/P11_2019_04abr29_v07_atribMOVPREFAE_etiq.csv')
p11_session4 = load_data('data/paciente11/P11_2019_05may13_v07_atribMOVPREFAE_etiq.csv')

#Dashboard ----------------------------------------------------------------------------------------------------
# Sidebar
st.sidebar.title("Seleccion de Paciente")
paciente_seleccionado = st.sidebar.selectbox(":hospital: Seleccionar paciente", ["Paciente 7", "Paciente 8", "Paciente 11"])


#Parte 1 ----------------------------------------
st.title(f'Presencia de sentiminetos - {paciente_seleccionado}')

col1, col2, col3, col4 = st.columns(4)
if col1.button(":spiral_calendar_pad: **Sesión 1**", help="Haz clic para ver la distribucion de las emociones de la sesión 1", use_container_width=True):
    if paciente_seleccionado == 'Paciente 7':
        data = p07_session1
    elif paciente_seleccionado == 'Paciente 8':
        data = p08_session1
    elif paciente_seleccionado == 'Paciente 11':
        data = p11_session1
        
    st.header('Tiempo de las Emociones - Sesion 1')

    # Identificar las columnas de sentimientos
    sentiment_columns = ['01_C', '02_A', '03_D', '04_M']

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 3))

    # Graficar cada sentimiento
    for i, col in enumerate(sentiment_columns):
        # Filtrar los intervalos de tiempo donde el sentimiento está presente (valor = 1)
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
        # Capturar el último intervalo si termina en el último valor de la columna
        if start is not None:
            intervals.append((start, data['time'].iloc[-1]))

        # Dibujar barras horizontales para los intervalos
        for (start, end) in intervals:
            ax.hlines(y=i, xmin=start, xmax=end, color=plt.get_cmap("Set1")(i), linewidth=6)

    # Etiquetas y formato
    ax.set_yticks(range(len(sentiment_columns)))
    ax.set_yticklabels(sentiment_columns)
    ax.set_xlabel("Tiempo")
    ax.set_title("Presencia de sentimientos en intervalos de tiempo")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    st.pyplot(fig)
    

if col2.button(":spiral_calendar_pad: **Sesión 2**", help="Haz clic para ver la distribucion de las emociones de la sesión 2", use_container_width=True):
    if paciente_seleccionado == 'Paciente 7':
        data = p07_session2
    elif paciente_seleccionado == 'Paciente 8':
        data = p08_session2
    elif paciente_seleccionado == 'Paciente 11':
        data = p11_session2
    
    st.header('Tiempo de las Emociones - Sesion 2')

    # Identificar las columnas de sentimientos
    sentiment_columns = ['01_C', '02_A', '03_D', '04_M']

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 3))

    # Graficar cada sentimiento
    for i, col in enumerate(sentiment_columns):
        # Filtrar los intervalos de tiempo donde el sentimiento está presente (valor = 1)
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
        # Capturar el último intervalo si termina en el último valor de la columna
        if start is not None:
            intervals.append((start, data['time'].iloc[-1]))

        # Dibujar barras horizontales para los intervalos
        for (start, end) in intervals:
            ax.hlines(y=i, xmin=start, xmax=end, color=plt.get_cmap("Set1")(i), linewidth=6)

    # Etiquetas y formato
    ax.set_yticks(range(len(sentiment_columns)))
    ax.set_yticklabels(sentiment_columns)
    ax.set_xlabel("Tiempo")
    ax.set_title("Presencia de sentimientos en intervalos de tiempo")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    st.pyplot(fig)

if col3.button(":spiral_calendar_pad: **Sesión 3**", help="Haz clic para ver la distribucion de las emociones de la sesión 3", use_container_width=True):
    if paciente_seleccionado == 'Paciente 7':
        data = p07_session3
    elif paciente_seleccionado == 'Paciente 8':
        data = p08_session3
    elif paciente_seleccionado == 'Paciente 11':
        data = p11_session3

    st.header('Tiempo de las Emociones - Sesion 3')

    # Identificar las columnas de sentimientos
    sentiment_columns = ['01_C', '02_A', '03_D', '04_M']

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 3))

    # Graficar cada sentimiento
    for i, col in enumerate(sentiment_columns):
        # Filtrar los intervalos de tiempo donde el sentimiento está presente (valor = 1)
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
        # Capturar el último intervalo si termina en el último valor de la columna
        if start is not None:
            intervals.append((start, data['time'].iloc[-1]))

        # Dibujar barras horizontales para los intervalos
        for (start, end) in intervals:
            ax.hlines(y=i, xmin=start, xmax=end, color=plt.get_cmap("Set1")(i), linewidth=6)

    # Etiquetas y formato
    ax.set_yticks(range(len(sentiment_columns)))
    ax.set_yticklabels(sentiment_columns)
    ax.set_xlabel("Tiempo")
    ax.set_title("Presencia de sentimientos en intervalos de tiempo")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    st.pyplot(fig)
    

if col4.button(":spiral_calendar_pad: **Sesión 4**", help="Haz clic para ver la distribucion de las emociones de la sesión 4", use_container_width=True):
    if paciente_seleccionado == 'Paciente 7':
        data = p07_session4
    elif paciente_seleccionado == 'Paciente 8':
        data = p08_session4
    elif paciente_seleccionado == 'Paciente 11':
        data = p11_session4
        
    st.header('Tiempo de las Emociones - Sesion 4')

    # Identificar las columnas de sentimientos
    sentiment_columns = ['01_C', '02_A', '03_D', '04_M']

    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(10, 3))

    # Graficar cada sentimiento
    for i, col in enumerate(sentiment_columns):
        # Filtrar los intervalos de tiempo donde el sentimiento está presente (valor = 1)
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
        # Capturar el último intervalo si termina en el último valor de la columna
        if start is not None:
            intervals.append((start, data['time'].iloc[-1]))

        # Dibujar barras horizontales para los intervalos
        for (start, end) in intervals:
            ax.hlines(y=i, xmin=start, xmax=end, color=plt.get_cmap("Set1")(i), linewidth=6)

    # Etiquetas y formato
    ax.set_yticks(range(len(sentiment_columns)))
    ax.set_yticklabels(sentiment_columns)
    ax.set_xlabel("Tiempo")
    ax.set_title("Presencia de sentimientos en intervalos de tiempo")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    st.pyplot(fig)

# Línea divisoria visual ----------------------------------------
st.markdown('---')
st.markdown('<hr style="border: 2px solid #CB1A2D; background-color: #3498db; height: 5px;">', unsafe_allow_html=True)

#Parte 2 ----------------------------------------
st.title(f'Distancias maximas por sesion - {paciente_seleccionado}')

if paciente_seleccionado == 'Paciente 7':
    dists_max = pd.DataFrame([p07_session1['Distancia'].sum(),p07_session2['Distancia'].sum(),p07_session3['Distancia'].sum(),p07_session4['Distancia'].sum()],columns=['DistMax'],index=['session1','session2','session3','session4'])
elif paciente_seleccionado == 'Paciente 8':
    dists_max = pd.DataFrame([p08_session1['Distancia'].sum(),p08_session2['Distancia'].sum(),p08_session3['Distancia'].sum(),p08_session4['Distancia'].sum()],columns=['DistMax'],index=['session1','session2','session3','session4'])
elif paciente_seleccionado == 'Paciente 11':
    dists_max = pd.DataFrame([p11_session1['Distancia'].sum(),p11_session2['Distancia'].sum(),p11_session3['Distancia'].sum(),p11_session4['Distancia'].sum()],columns=['DistMax'],index=['session1','session2','session3','session4'])

# Crear la figura y los ejes
fig, ax = plt.subplots(figsize=(10, 3))

# Crear la gráfica de barras
plt.style.use('classic')  
# Gráfica de barras
colors = plt.get_cmap("Set1").colors  # Usamos la paleta Set2 para colores suaves y equilibrados

bars = plt.bar(dists_max.index, dists_max['DistMax'], color=colors[:len(dists_max)])
# Rotar las etiquetas del eje X
plt.xticks(rotation=45)

# Añadir etiquetas y título
plt.xlabel("Sesiones", fontsize=12)
plt.ylabel("Distancia Maxima", fontsize=12)
plt.title(f"Distancias Maximas de cada Sesion - {paciente_seleccionado}", fontsize=14)

# Mostrar el valor encima de cada barra
for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f'{bar.get_height():.1f}',
        ha='center', va='bottom'
    )

st.pyplot(fig)
