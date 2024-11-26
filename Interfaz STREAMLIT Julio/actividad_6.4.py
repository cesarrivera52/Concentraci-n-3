import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import math 


st.set_page_config(page_title='Análisis socioformador', layout='wide')

st.title("Actividad 6.4")
st.header("Julio Sotero A01656310")
st.subheader("En esta actividad el propósito es la interacción con el framework Streamlit")

file_path = "P11.csv"
try:
    df = pd.read_csv(file_path)
    
    #Este código lo hicimos con el profe Juan Manuel, es para añadirle la columna de "time" y así poder graficar contra el tiempo (depende de la sesión, en este caso de 3)
    df["time"] = df.index * 3
    column_to_move = "time"
    columns = [column_to_move] + [col for col in df.columns if col != column_to_move]
    df = df[columns]


    analisis_columnas = ['Veloc', 'Acele', 'Des_x', 'Des_y']
    available_columns = [col for col in analisis_columnas if col in df.columns]
    
    if not available_columns:
        st.error("Las columnas para el análisis no están presentes en el archivo.")
    else:
        
        st.sidebar.header("Sidebar")
        x_columna = st.sidebar.selectbox("Selecciona una columna para graficar contra el tiempo ⏰ :", options=df.columns[1:])
        analisis_columna = st.sidebar.selectbox(
            "Selecciona una de las 4 columnas para desplegar sus gráficos 📈 :", 
            options=available_columns
        )
        
        st.header(f"Gráfica de {x_columna} vs tiempo ⏰")
        st.line_chart(df.set_index("time")[[x_columna]])
       
        st.header(f"Análisis de la columna: {analisis_columna} 📈 ")

        
        st.subheader("Gráfica de Barras")
        sampled_df = df[[analisis_columna]].iloc[::50] #muestreo para que no salgan tan amontanadas   
        st.bar_chart(sampled_df)
       
        st.subheader("Histograma")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df[analisis_columna], bins=15, color='blue', alpha=0.7)
        ax.set_title(f"Histograma de {analisis_columna}")
        ax.set_xlabel(analisis_columna)
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig) 
       
        st.subheader("Gráfica Densidad")
        st.area_chart(df[[analisis_columna]])
        
        st.subheader("Gráfica Alejada")
        st.scatter_chart(df[[analisis_columna]])

        #Este código lo hicimos con el profe Juan Manuel, es para sacar la distancia 
    def distance(x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)

    if {'Des_x', 'Des_y', 'Des_z'}.issubset(df.columns):
        df['Distancia'] = df.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
    else:
        st.error("Las columnas 'Des_x', 'Des_y', y 'Des_z' necesarias para calcular la distancia no están presentes.")
        st.stop()
       

 
    st.sidebar.header("Gráficas de Distancia")
    selected_plots = st.sidebar.multiselect(
        "Seleccione los gráficos a mostrar 📏 :",
        options=["Líneas", "Barras", "Histograma", "Densidad"],
        default=["Líneas", "Histograma"]
    )

    
    st.header("Análisis de la Distancia 📏")
    
 
    if "Líneas" in selected_plots:
        st.subheader("Gráfica de Líneas")
        st.line_chart(df.set_index("time")["Distancia"])


    if "Barras" in selected_plots:
        st.subheader("Gráfica de Barras")
        sampled_df = df[['time', 'Distancia']].iloc[::50]  
        st.bar_chart(sampled_df.set_index("time"))

  
    if "Histograma" in selected_plots:
        st.subheader("Histograma")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df['Distancia'], bins=15, color='blue', edgecolor='red', alpha=0.7)
        ax.set_title("Histograma de Distancia")
        ax.set_xlabel("Distancia")
        ax.set_ylabel("Frecuencia")
        st.pyplot(fig)


    if "Densidad" in selected_plots:
        st.subheader("Densidad")
        st.area_chart(df[['Distancia']])

     #Este código lo hicimos con el profe Juan Manuel, primero reemplazamos por 0 para que no haya valores negativos en las emociones 
    df = df.replace({-1: 0})
    
    column_sums = df.drop(columns=["time"]).sum()
  
    st.header("Gráfica de la Suma de Cada Columna ➕")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plt.style.use('seaborn-v0_8')
    colors = plt.get_cmap("Set1").colors  
    
    bars = ax.bar(column_sums.index, column_sums.values, color=colors[:len(column_sums)])
  
    ax.set_xticks(range(len(column_sums.index)))
    ax.set_xticklabels(column_sums.index, rotation=45)

    ax.set_xlabel("Columnas", fontsize=12)
    ax.set_ylabel("Suma", fontsize=12)
    ax.set_title("Suma de cada columna en el DataFrame", fontsize=14)
    
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f'{bar.get_height():.1f}',
            ha='center', va='bottom'
        )

    st.pyplot(fig)

     #Este código lo hicimos con el profe Juan Manuel, donde filtramos los intervalos donde el sentimiento esta presente (valor = 1)
    sentiment_columns = ['01_C', '02_A', '03_D', '04_M']

    st.header("Gráfica de presencia de sentimientos en intervalos de tiempo 🥹")
   
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 12))
 
    for i, ax in enumerate(axs):      
        col = sentiment_columns[i]
        intervals = []
        start = None
       
        for j in range(len(df)):
            if df[col].iloc[j] == 1:
                if start is None:
                    start = df['time'].iloc[j]
            else:
                if start is not None:
                    end = df['time'].iloc[j - 1]
                    intervals.append((start, end))
                    start = None     
        if start is not None:
            intervals.append((start, df['time'].iloc[-1]))
       
        for (start, end) in intervals:
            ax.hlines(y=i, xmin=start, xmax=end, color=plt.get_cmap("Set1")(i), linewidth=6)
       
        ax.set_yticks([i])
        ax.set_yticklabels([col])
        ax.set_xlabel("Tiempo")
        ax.set_title(f"Presencia de {col} en intervalos de tiempo")
        ax.grid(axis='x', linestyle='--', alpha=0.7)


    plt.tight_layout()
   
    st.pyplot(fig)

except FileNotFoundError:
    st.error(f"El archivo '{file_path}' no se encontró. Por favor, verifica la ruta.")


