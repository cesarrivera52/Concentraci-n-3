import streamlit as st
import pandas as pd
import math
import numpy as np 
import matplotlib.pyplot as plt

@st.cache_data
def distance(x, y, z):
  return math.sqrt(x**2 + y**2 + z**2)

df_p7s1 = pd.read_csv('P07_Sesion1.csv')
df_p7s1["time"] = df_p7s1.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p7s1.columns):
  df_p7s1['Distancia'] = df_p7s1.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()
df_p7s2 = pd.read_csv('P07_Sesion2.csv')
df_p7s2["time"] = df_p7s2.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p7s2.columns):
  df_p7s2['Distancia'] = df_p7s2.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()
df_p7s3 = pd.read_csv('P07_Sesion3.csv')
df_p7s3["time"] = df_p7s3.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p7s3.columns):
  df_p7s3['Distancia'] = df_p7s3.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()
df_p7s4 = pd.read_csv('P07_Sesion4.csv')
df_p7s4["time"] = df_p7s4.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p7s4.columns):
  df_p7s4['Distancia'] = df_p7s4.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()
df_p8s1 = pd.read_csv('P08_Sesion1.csv')
df_p8s1["time"] = df_p8s1.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p8s1.columns):
  df_p8s1['Distancia'] = df_p8s1.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()
df_p8s2 = pd.read_csv('P08_Sesion2.csv')
df_p8s2["time"] = df_p8s2.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p8s2.columns):
  df_p8s2['Distancia'] = df_p8s2.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()

df_p8s3 = pd.read_csv('P08_Sesion3.csv')
df_p8s3["time"] = df_p8s3.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p8s3.columns):
  df_p8s3['Distancia'] = df_p8s3.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()

df_p8s4 = pd.read_csv('P08_Sesion4.csv')
df_p8s4["time"] = df_p8s4.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p8s4.columns):
  df_p8s4['Distancia'] = df_p8s4.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()

df_p11s1 = pd.read_csv('P11_Sesion1.csv')
df_p11s1["time"] = df_p11s1.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p11s1.columns):
  df_p11s1['Distancia'] = df_p11s1.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()

df_p11s2 = pd.read_csv('P11_Sesion2.csv')
df_p11s2["time"] = df_p11s2.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p11s2.columns):
  df_p11s2['Distancia'] = df_p11s2.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()

df_p11s3 = pd.read_csv('P11_Sesion3.csv')
df_p11s3["time"] = df_p11s3.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p11s3.columns):
  df_p11s3['Distancia'] = df_p11s3.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()

df_p11s4 = pd.read_csv('P11_Sesion4.csv')
df_p11s4["time"] = df_p11s4.index * 3
if {'Des_x', 'Des_y', 'Des_z'}.issubset(df_p11s4.columns):
  df_p11s4['Distancia'] = df_p11s4.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
else:
  st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
  st.stop()
paciente7=[]
paciente7.append(df_p7s1)
paciente7.append(df_p7s2)
paciente7.append(df_p7s3)
paciente7.append(df_p7s4)
paciente8=[]
paciente8.append(df_p8s1)
paciente8.append(df_p8s2)
paciente8.append(df_p8s3)
paciente8.append(df_p8s4)
paciente11=[]
paciente11.append(df_p11s1)
paciente11.append(df_p11s2)
paciente11.append(df_p11s3)
paciente11.append(df_p11s4)
sesiones=['Sesion 1','Sesion 2','Sesion 3','Sesion 4']
st.sidebar.subheader('Seleccione un paciente')
var=st.sidebar.selectbox(label='Pacientes', options=['Paciente 7', 'Paciente 8', 'Paciente 11'])

if 'Paciente 7' in var:
  st.title('Distancia recorrida Paciente 7')
  valores=[]
  for i in range(len(paciente7)):
    valores.append(sum(paciente7[i]['Distancia']))
  fig1, ax1 = plt.subplots()
  ax1.bar(sesiones,valores)
  ax1.set_title('Distancia recorrida por sesión')
  ax1.set_xlabel('Sesiones')
  ax1.set_ylabel('Distancia recorrida')
  st.pyplot(fig1)
if 'Paciente 8' in var:
  st.title('Distancia recorrida Paciente 8')
  valores=[]
  for i in range(len(paciente8)):
    valores.append(sum(paciente8[i]['Distancia']))
  fig2, ax2 = plt.subplots()
  ax2.bar(sesiones,valores)
  ax2.set_title('Distancia recorrida por sesión')
  ax2.set_xlabel('Sesiones')
  ax2.set_ylabel('Distancia recorrida')
  st.pyplot(fig2)
if 'Paciente 11' in var:
  st.title('Distancia recorrida Paciente 11')
  valores=[]
  for i in range(len(paciente11)):
    valores.append(sum(paciente11[i]['Distancia']))
  fig3, ax3 = plt.subplots()
  ax3.bar(sesiones,valores)
  ax3.set_title('Distancia recorrida por sesión')
  ax3.set_xlabel('Sesiones')
  ax3.set_ylabel('Distancia recorrida')
  st.pyplot(fig3)


col1, col2, col3, col4 = st.columns(4)

if col1.button(":spiral_calendar_pad: **Sesión 1**", help="Haz clic para ver la gráfica de emociones de la sesión 1", use_container_width=True):
    if var == "Paciente 7":
        data = df_p7s1
    elif var == "Paciente 8":
        data = df_p8s1
    elif var == "Paciente 11":
        data = df_p11s1

    st.header("Presencia de sentimientos - Sesión 1")
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
    ax.set_xlabel("Tiempo")
    ax.set_title("Presencia de sentimientos en intervalos de tiempo")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

if col2.button(":spiral_calendar_pad: **Sesión 2**", help="Haz clic para ver la gráfica de emociones de la sesión 2", use_container_width=True):
    if var == "Paciente 7":
        data = df_p7s2
    elif var == "Paciente 8":
        data = df_p8s2
    elif var == "Paciente 11":
        data = df_p11s2

    st.header("Presencia de sentimientos - Sesión 2")
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
    ax.set_xlabel("Tiempo")
    ax.set_title("Presencia de sentimientos en intervalos de tiempo")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

if col3.button(":spiral_calendar_pad: **Sesión 3**", help="Haz clic para ver la gráfica de emociones de la sesión 3", use_container_width=True):
    if var == "Paciente 7":
        data = df_p7s3
    elif var == "Paciente 8":
        data = df_p8s3
    elif var == "Paciente 11":
        data = df_p11s3

    st.header("Presencia de sentimientos - Sesión 3")
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
    ax.set_xlabel("Tiempo")
    ax.set_title("Presencia de sentimientos en intervalos de tiempo")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)

if col4.button(":spiral_calendar_pad: **Sesión 4**", help="Haz clic para ver la gráfica de emociones de la sesión 4", use_container_width=True):
    if var == "Paciente 7":
        data = df_p7s4
    elif var == "Paciente 8":
        data = df_p8s4
    elif var == "Paciente 11":
        data = df_p11s4

    st.header("Presencia de sentimientos - Sesión 4")
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
    ax.set_xlabel("Tiempo")
    ax.set_title("Presencia de sentimientos en intervalos de tiempo")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
