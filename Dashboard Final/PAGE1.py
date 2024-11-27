import streamlit as st
import datetime
import numpy as np
from datetime import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
st.markdown("""  
<style>  
.stMainBlockContainer{
    background-color: #000000;
}
.stAppHeader{
    background-color: #000000;
}
</style>  
""", unsafe_allow_html=True)  

plt.style.use('dark_background')
#@st.cache
PATIENT=st.sidebar.selectbox(label='Eliga un paciente:',options=['Paciente 1','Paciente 2','Paciente 3'])
VENTANEO=st.sidebar.selectbox(label='Seleccione el ventaneo',options=['Cada 3 segundos','Cada 5 segundos','Cada 7 segundos','Cada 9 segundos','Cada 11 segundos'])
if 'Cada 3 segundos' in VENTANEO:
    VENTANEO_SELECT=3
elif 'Cada 5 segundos' in VENTANEO:
    VENTANEO_SELECT=5
elif 'Cada 7 segundos' in VENTANEO:
    VENTANEO_SELECT=7
elif 'Cada 9 segundos' in VENTANEO:
    VENTANEO_SELECT=9
elif 'Cada 11 segundos' in VENTANEO:
    VENTANEO_SELECT=11
def times(data):
    data['Tiempo']=data.index*VENTANEO_SELECT
    data.dropna(inplace=True)
    data['Tiempo']=data['Tiempo'].apply(lambda x: datetime.timedelta(seconds=x).total_seconds() / 60)
    return data
def filter(minn,maxx):
    if 'Paciente 1' in PATIENT:
        from config import df1p1
        from config import df2p1
        from config import df3p1
        from config import df4p1
        df1 = times(df1p1)
        df2 = times(df2p1)
        df3 = times(df3p1)
        df4 = times(df4p1)
    elif 'Paciente 2' in PATIENT:
        from config import df1p2
        from config import df2p2
        from config import df3p2
        from config import df4p2
        df1 = times(df1p2)
        df2 = times(df2p2)
        df3 = times(df3p2)
        df4 = times(df4p2)
    elif 'Paciente 3' in PATIENT:
        from config import df1p3
        from config import df2p3
        from config import df3p3
        from config import df4p3
        df1 = times(df1p3)
        df2 = times(df2p3)
        df3 = times(df3p3)
        df4 = times(df4p3)
    df1= df1[(df1['Tiempo'] >= minn) & (df1['Tiempo'] <= maxx)]
    df2 = df2[(df2['Tiempo'] >= minn) & (df2['Tiempo'] <= maxx)]
    df3 = df3[(df3['Tiempo'] >= minn) & (df3['Tiempo'] <= maxx)]
    df4 = df4[(df4['Tiempo'] >= minn) & (df4['Tiempo'] <= maxx)]
    return df1,df2,df3,df4
#st.set_page_config(page_title='Análisis de datos',layout="wide")
#st.title('Análisis de datos')
#tab1,tab2=st.tabs(['Análisis de variables en el tiempo','Análisis comparativo'])
#st.sidebar.title('Panel de control')
st.sidebar.empty()
df1,df2,df3,df4=filter(0,1e70)
lista=[df1,df2,df3,df4]
maxim=[]
for df in lista:
    maxim.append(max(df['Tiempo']))
SLIDER = st.sidebar.slider("Seleccione un rango de tiempo:", value=(0.0,min(maxim)))
st.sidebar.write("Tiempo seleccionado:\n", str(round(SLIDER[1]-SLIDER[0],2))+' '+'min')
df1,df2,df3,df4=filter(float(SLIDER[0]),float(SLIDER[1]))

import streamlit as st

col1, col2 = st.columns(2)

SECTION=st.sidebar.selectbox(label='Grafique en función del tiempo:',options=['Distancia','Velocidad','Aceleración','Presión','Velocidad de la presión','Aceleración de la presión'])
import matplotlib.pyplot as plt
# Asumiendo que df1, df2, df3 y df4 son DataFrames ya definidos
sesiones = [df1, df2, df3, df4]
list_sesiones,intervalos_main,emoc=[],[],[]
sentiment_columns = ['01_C', '02_A', '03_D', '04_M']
# Creo la figura con 4 subplots en 2 filas y 2 columnas
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
# Itero sobre cada sesión y su subplot
for s, ax in enumerate(axs.flatten()):
    # Sesión actual
    session_data = sesiones[s]
    # Grafico cada columna de sentimiento en el subplot correspondiente
    for i, col in enumerate(sentiment_columns):
        # Filtro los intervalos de tiempo donde el sentimiento está presente (valor = 1)
        intervalos = []
        start = None
        for j in range(len(session_data)):
            if session_data[col].iloc[j] == 1:
                if start is None:
                    start = session_data['Tiempo'].iloc[j]
            else:
                if start is not None:
                    end = session_data['Tiempo'].iloc[j - 1]
                    intervalos.append((start, end))
                    list_sesiones.append(s+1)
                    intervalos_main.append((start, end))
                    emoc.append(col)
                    start = None
        # Capturar el último intervalo si termina en el último valor de la columna
        if start is not None:
            intervalos.append((start, session_data['Tiempo'].iloc[-1]))
            list_sesiones.append(s+1)
            intervalos_main.append((start, session_data['Tiempo'].iloc[-1]))
            emoc.append(col)

        # Dibujar barras horizontales para los intervalos
        for (start, end) in intervalos:
            ax.hlines(y=i, xmin=start, xmax=end, color=plt.get_cmap("Set3_r")(i), linewidth=6)
    # Etiquetas y formato
    ax.set_yticks(range(len(sentiment_columns)))
    ax.set_yticklabels(sentiment_columns)
    ax.set_xlabel("Tiempo")
    ax.set_title(f"Presencia de sentimientos en intervalos de tiempo - Sesión {s + 1}")
    ax.grid(axis='x', linestyle='--', alpha=0.7)
maximos,etiquetas=[],[]
con=0
for i,j in intervalos_main:
    maximos.append(round(j-i,2))
    etiquetas.append(list_sesiones[con])
    con+=1
data_mini=pd.DataFrame()
data_mini['sesion']=etiquetas
data_mini['valor']=maximos
data_mini['emocion']=emoc
data_mini['emocion']= data_mini['emocion'].replace({'01_C':'Cansancio'})
data_mini['emocion']= data_mini['emocion'].replace({'02_A':'Ansiedad'})
data_mini['emocion']= data_mini['emocion'].replace({'03_D':'Dolor'})
data_mini['emocion']= data_mini['emocion'].replace({'04_M':'Motivación'})
    
#plt.tight_layout()
with col1: 
    st.pyplot(fig)
    # Función para calcular cruces por cero
    def calculate_zero_crossings(df):
        numeric_cols = df.select_dtypes(include=['number']).columns
        scaler = StandardScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols)
        zero_crossings = {
            col: ((df_normalized[col].shift(1) * df_normalized[col]) < 0).sum()
            for col in df_normalized.columns
        }
        return pd.DataFrame(list(zero_crossings.items()), columns=['Column', 'Zero_Crossings'])

    # Crear un radar chart para múltiples sesiones
    def create_combined_radar_chart(radar_data, labels, title):
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

        # Construir ángulos para las variables (mismas para todos los DataFrames)
        categories = radar_data[0]['Column']
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el radar

        # Dibujar cada serie de datos en el radar
        i=0
        for session_data, label in zip(radar_data, labels):
            values = session_data['Zero_Crossings'].tolist()
            values += values[:1]  # Cerrar el gráfico
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=label,color=plt.get_cmap("Set3_r")(i+1))
            ax.fill(angles, values, alpha=0.25,color=plt.get_cmap("Set3_r")(i+1))
            i+=1

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, max(max(df['Zero_Crossings']) for df in radar_data))
        ax.set_title(title, size=15, color='white', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        return fig

    # Simular DataFrames df1, df2, df3, df4
    dff1=df1.drop(columns=['F5EEX','F6EES','F8EIX','F7EIS','F9EEX','F10ES','F11EX','F12ES','F13EV','F14EV','F17MS','F18MX','F15MS','F16MX','F19MH','F20MH','F1EBX','F2EBS','F4EBX','F3EBS','01_C','02_A','03_D','04_M','Tiempo'])
    dff2=df2.drop(columns=['F5EEX','F6EES','F8EIX','F7EIS','F9EEX','F10ES','F11EX','F12ES','F13EV','F14EV','F17MS','F18MX','F15MS','F16MX','F19MH','F20MH','F1EBX','F2EBS','F4EBX','F3EBS','01_C','02_A','03_D','04_M','Tiempo'])
    dff3=df3.drop(columns=['F5EEX','F6EES','F8EIX','F7EIS','F9EEX','F10ES','F11EX','F12ES','F13EV','F14EV','F17MS','F18MX','F15MS','F16MX','F19MH','F20MH','F1EBX','F2EBS','F4EBX','F3EBS','01_C','02_A','03_D','04_M','Tiempo'])
    dff4=df4.drop(columns=['F5EEX','F6EES','F8EIX','F7EIS','F9EEX','F10ES','F11EX','F12ES','F13EV','F14EV','F17MS','F18MX','F15MS','F16MX','F19MH','F20MH','F1EBX','F2EBS','F4EBX','F3EBS','01_C','02_A','03_D','04_M','Tiempo'])

    
    lista2=[dff1,dff2,dff3,dff4]
    # Calcular cruces por cero para cada sesión
    radar_data = [calculate_zero_crossings(df) for df in lista2]
    combined_radar_chart1 = create_combined_radar_chart(
        radar_data, 
        labels=["Sesión 1", "Sesión 2", "Sesión 3", "Sesión 4"],
        title="Cruces por Cero - Variables dinámicas"
    )
    # Simular DataFrames df1, df2, df3, df4
    dff1=df1.drop(columns=['Veloc','Acele','Des_x','Des_y','Des_z','Presn','VelPr','AcePr','01_C','02_A','03_D','04_M','Tiempo','Distancia'])
    dff2=df2.drop(columns=['Veloc','Acele','Des_x','Des_y','Des_z','Presn','VelPr','AcePr','01_C','02_A','03_D','04_M','Tiempo','Distancia'])
    dff3=df3.drop(columns=['Veloc','Acele','Des_x','Des_y','Des_z','Presn','VelPr','AcePr','01_C','02_A','03_D','04_M','Tiempo','Distancia'])
    dff4=df4.drop(columns=['Veloc','Acele','Des_x','Des_y','Des_z','Presn','VelPr','AcePr','01_C','02_A','03_D','04_M','Tiempo','Distancia'])

    
    lista2=[dff1,dff2,dff3,dff4]
    # Calcular cruces por cero para cada sesión
    radar_data = [calculate_zero_crossings(df) for df in lista2]
    combined_radar_chart2 = create_combined_radar_chart(
        radar_data, 
        labels=["Sesión 1", "Sesión 2", "Sesión 3", "Sesión 4"],
        title="Cruces por Cero - Líneas de la cara"
    )
    # Crear un mapa de calor para cada sesión
    def create_heatmap(df,df2,df3,df4, session_label1,session_label2,session_label3,session_label4):
        fig, axs = plt.subplots(2, 2, figsize=(14, 11.5))
        numeric_cols = df.select_dtypes(include=['number']).columns
        correlation_matrix = df[numeric_cols].corr()
        numeric_cols = df2.select_dtypes(include=['number']).columns
        correlation_matrix2 = df2[numeric_cols].corr()
        numeric_cols = df3.select_dtypes(include=['number']).columns
        correlation_matrix3 = df3[numeric_cols].corr()
        numeric_cols = df4.select_dtypes(include=['number']).columns
        correlation_matrix4 = df4[numeric_cols].corr()
        # Mapa de calor para el primer DataFrame ,color=plt.get_cmap("Set3_r")(i+1)
        sns.heatmap(correlation_matrix, ax=axs[0,0], cmap='YlGnBu', cbar=True) 
        axs[0,0].set_title(session_label1) 
        # Mapa de calor para el segundo DataFrame 
        sns.heatmap(correlation_matrix2, ax=axs[0,1], cmap='YlGnBu', cbar=True) 
        axs[0,1].set_title(session_label2)

        sns.heatmap(correlation_matrix3, ax=axs[1,0], cmap='YlGnBu', cbar=True) 
        axs[1,0].set_title(session_label3)

        sns.heatmap(correlation_matrix4, ax=axs[1,1], cmap='YlGnBu', cbar=True) 
        axs[1,1].set_title(session_label4)
        #plt.tight_layout()
        return fig

    # Crear los gráficos
    heatmap1 = create_heatmap(df1,df2,df3,df4, "Sesión 1",'Sesión 2','Sesión 3','Sesión 4')
    # Datos para cada sección
    s1c=data_mini[(data_mini['emocion']=='Cansancio') & (data_mini['sesion']==1)]
    s1a=data_mini[(data_mini['emocion']=='Ansiedad') & (data_mini['sesion']==1)]
    s1d=data_mini[(data_mini['emocion']=='Dolor') & (data_mini['sesion']==1)]
    s1m=data_mini[(data_mini['emocion']=='Motivación') & (data_mini['sesion']==1)]
    seccion1 = [np.where(np.isnan(s1c['valor'].max()),0,s1c['valor'].max()), np.where(np.isnan(s1a['valor'].max()),0,s1a['valor'].max()), np.where(np.isnan(s1d['valor'].max()),0,s1d['valor'].max()),np.where(np.isnan(s1m['valor'].max()),0,s1m['valor'].max())]
    
    s1c=data_mini[(data_mini['emocion']=='Cansancio') & (data_mini['sesion']==2)]
    s1a=data_mini[(data_mini['emocion']=='Ansiedad') & (data_mini['sesion']==2)]
    s1d=data_mini[(data_mini['emocion']=='Dolor') & (data_mini['sesion']==2)]
    s1m=data_mini[(data_mini['emocion']=='Motivación') & (data_mini['sesion']==2)]
    seccion2 = [np.where(np.isnan(s1c['valor'].max()),0,s1c['valor'].max()), np.where(np.isnan(s1a['valor'].max()),0,s1a['valor'].max()), np.where(np.isnan(s1d['valor'].max()),0,s1d['valor'].max()),np.where(np.isnan(s1m['valor'].max()),0,s1m['valor'].max())]

    s1c=data_mini[(data_mini['emocion']=='Cansancio') & (data_mini['sesion']==3)]
    s1a=data_mini[(data_mini['emocion']=='Ansiedad') & (data_mini['sesion']==3)]
    s1d=data_mini[(data_mini['emocion']=='Dolor') & (data_mini['sesion']==3)]
    s1m=data_mini[(data_mini['emocion']=='Motivación') & (data_mini['sesion']==3)]
    seccion3 = [np.where(np.isnan(s1c['valor'].max()),0,s1c['valor'].max()), np.where(np.isnan(s1a['valor'].max()),0,s1a['valor'].max()), np.where(np.isnan(s1d['valor'].max()),0,s1d['valor'].max()),np.where(np.isnan(s1m['valor'].max()),0,s1m['valor'].max())]

    s1c=data_mini[(data_mini['emocion']=='Cansancio') & (data_mini['sesion']==4)]
    s1a=data_mini[(data_mini['emocion']=='Ansiedad') & (data_mini['sesion']==4)]
    s1d=data_mini[(data_mini['emocion']=='Dolor') & (data_mini['sesion']==4)]
    s1m=data_mini[(data_mini['emocion']=='Motivación') & (data_mini['sesion']==4)]
    seccion4 = [np.where(np.isnan(s1c['valor'].max()),0,s1c['valor'].max()), np.where(np.isnan(s1a['valor'].max()),0,s1a['valor'].max()), np.where(np.isnan(s1d['valor'].max()),0,s1d['valor'].max()),np.where(np.isnan(s1m['valor'].max()),0,s1m['valor'].max())]
    

    # Etiquetas para el eje x
    labels = ['Cansancio', 'Ansiedad', 'Dolor', 'Motivación']

    # Crear subplots
    fig3, axs3 = plt.subplots(1, 4, figsize=(18, 5))

    # Gráfico para la sección 1
    axs3[0].bar(labels, seccion1, color=plt.get_cmap("Set3_r")(1))
    axs3[0].set_title('Sesión 1')
    axs3[0].set_ylabel('Tiempo Máximo')

    # Gráfico para la sección 2
    axs3[1].bar(labels, seccion2, color=plt.get_cmap("Set3_r")(3))
    axs3[1].set_title('Sesión 2')
    axs3[1].set_ylabel('Tiempo Máximo')

    # Gráfico para la sección 3
    axs3[2].bar(labels, seccion3, color=plt.get_cmap("Set3_r")(4))
    axs3[2].set_title('Sesión 3')
    axs3[2].set_ylabel('Tiempo Máximo')

    # Gráfico para la sección 4
    axs3[3].bar(labels, seccion4, color=plt.get_cmap("Set3_r")(7))
    axs3[3].set_title('Sesión 4')
    axs3[3].set_ylabel('Tiempo Máximo')
    col3,col4=st.columns(2)
    with col3:
        st.pyplot(combined_radar_chart1)
    with col4:
        st.pyplot(combined_radar_chart2)
    st.pyplot(fig3)
with col2:
    if 'Distancia' in SECTION:
        var='Distancia'
        var2='Distancia'
    if 'Velocidad' in SECTION:
        var='Velocidad'
        var2='Veloc'
    if 'Aceleración' in SECTION:
        var='Aceleración'
        var2='Acele'
    if 'Presión' in SECTION:
        var='Presión'
        var2='Presn'
    if 'Velocidad de la presión' in SECTION:
        var='Velocidad de la presión'
        var2='VelPr'
    if 'Aceleración de la presión' in SECTION:
        var='Aceleración de la presión'
        var2='AcePr'
    fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.23))
    # Crear el gráfico de áreas
    plt.fill_between(df1['Tiempo'], df1[var2].cumsum(), color=plt.get_cmap("Set3_r")(1), alpha=0.5, label='Sesión 1')
    plt.fill_between(df2['Tiempo'], df2[var2].cumsum(), color=plt.get_cmap("Set3_r")(3), alpha=0.5, label='Sesión 2')
    plt.fill_between(df3['Tiempo'], df3[var2].cumsum(), color=plt.get_cmap("Set3_r")(4), alpha=0.5, label='Sesión 3')
    plt.fill_between(df4['Tiempo'], df4[var2].cumsum(), color=plt.get_cmap("Set3_r")(7), alpha=0.5, label='Sesión 4')

    # Etiquetas y título
    plt.xlabel('Tiempo de la sesión (min)')
    plt.ylabel(f'{var}')
    plt.title(f'{var} acumulada en el tiempo')
    plt.legend(loc='upper left')
    #plt.tight_layout()
    st.pyplot(fig2)
    import matplotlib.pyplot as plt

    

    # Ajustar el diseño
    #plt.tight_layout()
    st.pyplot(heatmap1)
    