import streamlit as st
import datetime
import numpy as np
from datetime import time
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

plt.style.use('dark_background')
def filter(minn,maxx):
    from config import df1
    from config import df2
    from config import df3
    from config import df4
    df1= df1[(df1['Tiempo'] >= minn) & (df1['Tiempo'] <= maxx)]
    df2 = df2[(df2['Tiempo'] >= minn) & (df2['Tiempo'] <= maxx)]
    df3 = df3[(df3['Tiempo'] >= minn) & (df3['Tiempo'] <= maxx)]
    df4 = df4[(df4['Tiempo'] >= minn) & (df4['Tiempo'] <= maxx)]
    return df1,df2,df3,df4

st.set_page_config(page_title='Análisis de datos',layout="wide")
#st.title('Análisis de datos')
############################## PESTAÑAS ##########################################################
tab1,tab2=st.tabs(['Análisis de variables en el tiempo','Análisis comparativo'])
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

############################## PESTAÑA 1 ##########################################################
with tab1:
    import streamlit as st

    col1, col2 = st.columns(2)

    SECTION=st.sidebar.selectbox(label='Grafique en función del tiempo:',options=['Distancia','Velocidad','Aceleración','Presión','Velocidad de la presión','Aceleración de la presión'])
    MULTI_SECTION1=st.sidebar.multiselect(label='Radar variables dinámicas:',options=['Sesión 1','Sesión 2','Sesión 3','Sesión 4'])
    MULTI_SECTION2=st.sidebar.multiselect(label='Radar líneas de la cara:',options=['Sesión 1','Sesión 2','Sesión 3','Sesión 4'])
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
                ax.hlines(y=i, xmin=start, xmax=end, color=plt.get_cmap("Set1")(i), linewidth=6)

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
        
    plt.tight_layout()
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
            for session_data, label in zip(radar_data, labels):
                values = session_data['Zero_Crossings'].tolist()
                values += values[:1]  # Cerrar el gráfico
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=label)
                ax.fill(angles, values, alpha=0.25)

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
        col3, col4 = st.columns(2)
        # Crear un mapa de calor para cada sesión
        def create_heatmap(df,df2, session_label1,session_label2):
            fig, axs = plt.subplots(1, 2, figsize=(15, 6))
            numeric_cols = df.select_dtypes(include=['number']).columns
            correlation_matrix = df[numeric_cols].corr()
            numeric_cols = df2.select_dtypes(include=['number']).columns
            correlation_matrix2 = df2[numeric_cols].corr()

            # Mapa de calor para el primer DataFrame 
            sns.heatmap(correlation_matrix, ax=axs[0], cmap='viridis', cbar=False) 
            axs[0].set_title(session_label1) 
            # Mapa de calor para el segundo DataFrame 
            sns.heatmap(correlation_matrix2, ax=axs[1], cmap='viridis', cbar=True) 
            axs[1].set_title(session_label2)
            return fig

        # Crear los gráficos
        heatmap1 = create_heatmap(df1,df2, "Sesión 1",'Sesión 2')
        heatmap2 = create_heatmap(df3,df4 ,"Sesión 3",'Sesión 4')
        
        with col3:
            st.pyplot(combined_radar_chart1)
            st.pyplot(heatmap1)

        with col4:
            st.pyplot(combined_radar_chart2)
            st.pyplot(heatmap2)

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
        fig2, axs2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6.23))
        # Crear el gráfico de áreas
        plt.fill_between(df1['Tiempo'], df1[var2].cumsum(), color='green', alpha=0.5, label='Sesión 1')
        plt.fill_between(df2['Tiempo'], df2[var2].cumsum(), color='lightblue', alpha=0.5, label='Sesión 2')
        plt.fill_between(df3['Tiempo'], df3[var2].cumsum(), color='blue', alpha=0.5, label='Sesión 3')
        plt.fill_between(df4['Tiempo'], df4[var2].cumsum(), color='purple', alpha=0.5, label='Sesión 4')

        # Etiquetas y título
        plt.xlabel('Tiempo de la sesión (min)')
        plt.ylabel(f'{var}')
        plt.title(f'{var} acumulada en el tiempo')
        plt.legend(loc='upper left')
        plt.tight_layout()
        st.pyplot(fig2)
        import matplotlib.pyplot as plt

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
        print(s1c)

        # Etiquetas para el eje x
        labels = ['Cansancio', 'Ansiedad', 'Dolor', 'Motivación']

        # Crear subplots
        fig3, axs3 = plt.subplots(2, 2, figsize=(10, 8))

        # Gráfico para la sección 1
        axs3[0, 0].bar(labels, seccion1, color='green')
        axs3[0, 0].set_title('Tiempo máximo en una emoción Sesión 1')
        axs3[0, 0].set_ylabel('Tiempo')

        # Gráfico para la sección 2
        axs3[0, 1].bar(labels, seccion2, color='lightblue')
        axs3[0, 1].set_title('Tiempo máximo en una emoción Sesión 2')
        axs3[0, 1].set_ylabel('Tiempo')

        # Gráfico para la sección 3
        axs3[1, 0].bar(labels, seccion3, color='blue')
        axs3[1, 0].set_title('Tiempo máximo en una emoción Sesión 3')
        axs3[1, 0].set_ylabel('Tiempo')

        # Gráfico para la sección 4
        axs3[1, 1].bar(labels, seccion4, color='purple')
        axs3[1, 1].set_title('Tiempo máximo en una emoción Sesión 4')
        axs3[1, 1].set_ylabel('Tiempo')

        # Ajustar el diseño
        plt.tight_layout()
        st.pyplot(fig3)

############################## PESTAÑA 2 ##########################################################
with tab2:
    def pir(dfg):
                # Filtros para derecho e izquierdo
        derecho = dfg[['F1EBX', 'F4EBX', 'F5EEX', 'F8EIX', 'F14EV', 'F9EEX', 'F11EX', 'F15MS', 'F19MH', 'F17MS']]
        izquierdo = dfg[['F2EBS', 'F3EBS', 'F6EES', 'F7EIS', 'F13EV', 'F12ES', 'F10ES', 'F16MX', 'F20MH', 'F18MX']]

        # Estandarización de las columnas numéricas
        scaler = StandardScaler()
        derecho_normalized = pd.DataFrame(scaler.fit_transform(derecho), columns=derecho.columns)
        izquierdo_normalized = pd.DataFrame(scaler.fit_transform(izquierdo), columns=izquierdo.columns)

        # Contar cruces por cero para cada columna
        cross_zero_counts_derecho = [
            ((derecho_normalized[col].shift(1) * derecho_normalized[col]) < 0).sum()
            for col in derecho_normalized.columns
        ]

        cross_zero_counts_izquierdo = [
            ((izquierdo_normalized[col].shift(1) * izquierdo_normalized[col]) < 0).sum()
            for col in izquierdo_normalized.columns
        ]

        # Crear un DataFrame con los valores de los cruces por cero
        df_paired = pd.DataFrame({
            'Columnas_Derecho': derecho.columns,
            'Columnas_Izquierdo': izquierdo.columns,
            'Derecho': cross_zero_counts_derecho,
            'Izquierdo': cross_zero_counts_izquierdo
        })

        # Agregar una fila para las sumatorias totales utilizando pd.concat()
        sumatoria_derecho = sum(df_paired['Derecho'])
        sumatoria_izquierdo = sum(df_paired['Izquierdo'])
        df_sumatoria = pd.DataFrame({
            'Columnas_Derecho': ['Sumatoria Total'],
            'Columnas_Izquierdo': ['Sumatoria Total'],
            'Derecho': [sumatoria_derecho],
            'Izquierdo': [sumatoria_izquierdo]
        })
        df_paired = pd.concat([df_paired], ignore_index=True)

        # Crear la gráfica de pirámide
        fig, ax = plt.subplots(figsize=(14, 8))

        # Posiciones en el eje y (una posición por cada par de columnas más la fila de sumatorias)
        y_positions = range(len(df_paired))

        # Dibujar las barras horizontales para cada par de columnas y la sumatoria
        ax.barh(y_positions, df_paired['Derecho'], color='skyblue', label='Derecho', align='center')
        ax.barh(y_positions, -df_paired['Izquierdo'], color='lightpink', label='Izquierdo', align='center')  # Negativo para enfrentar

        # Configuración del eje y con etiquetas a ambos lados
        ax.set_yticks(y_positions)
        ax.set_yticklabels([''] * len(df_paired))  # Ocultamos temporalmente las etiquetas centrales
        ax.invert_yaxis()  # Para que las barras más grandes queden en la parte superior

        # Etiquetas a la izquierda y derecha de las barras
        for i, (col_derecho, col_izquierdo) in enumerate(zip(df_paired['Columnas_Derecho'], df_paired['Columnas_Izquierdo'])):
            ax.text(-max(df_paired['Izquierdo']) * 1.1, i, col_izquierdo, ha='right', va='center', fontsize=20, color='white')  # Etiquetas de izquierdo
            ax.text(max(df_paired['Derecho']) * 1.1, i, col_derecho, ha='left', va='center', fontsize=20, color='white')  # Etiquetas de derecho

        # Configuración de etiquetas y leyenda
        ax.set_xlabel('Zero Crossings')
        ax.set_title('Comparación de Cruces por Cero: Derecho vs Izquierdo por Pares (Incluyendo Sumatoria)')
        ax.legend(loc='upper right')

        # Línea central para separar los lados
        ax.axvline(0, color='black', linewidth=0.8)

        # Ajustar diseño y mostrar la gráfica
        plt.tight_layout()
        plt.show()
        return fig, df_sumatoria
    
    def boca_multiple_sesions(df_sesions):
        boc = df_sesions[['F15MS', 'F16MX', 'F17MS', 'F18MX', 'F19MH', 'F20MH']]
        # Estandarización de las columnas numéricas
        numeric_cols = boc.select_dtypes(include=['number']).columns
        scaler = StandardScaler()
        boc_normalized = pd.DataFrame(scaler.fit_transform(boc[numeric_cols]), columns=numeric_cols)

        # Contar cruces por cero para cada columna
        cross_zero_counts = {}
        for col in boc_normalized.columns:
            cross_zero_counts[col] = ((boc_normalized[col].shift(1) * boc_normalized[col]) < 0).sum()

        # Convertir el resultado en un DataFrame
        df = pd.DataFrame(list(cross_zero_counts.items()), columns=['Column', 'Zero_Crossings'])
        df = df.set_index('Column')

        # Ordenar los datos en orden descendente por 'Zero_Crossings'
        df_sorted = df.sort_values(by='Zero_Crossings', ascending=False).reset_index()

        # Crear el gráfico de área y la línea
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.fill_between(
            df_sorted['Column'],
            df_sorted['Zero_Crossings'],
            color="skyblue",
            alpha=0.4,
            label="Zero Crossings"
        )
       
        # Configuración de ejes y leyendas
        ax.set_xlabel('Column')
        ax.set_title('Area Chart')
        ax.tick_params(axis='x', rotation=45)
        ax.legend()
        plt.tight_layout()

    # Retornar el gráfico para mostrarlo en Streamlit
        return fig
    
    def ojos(dfg):
        ojos=dfg[['F5EEX','F6EES','F7EIS','F8EIX','F9EEX','F10ES','F11EX','F12ES','F13EV','F14EV']]
        

            # Estandarización de las columnas numéricas
        numeric_cols = ojos.select_dtypes(include=['number']).columns
        scaler = StandardScaler()
        boc_normalized = pd.DataFrame(scaler.fit_transform(ojos[numeric_cols]), columns=numeric_cols)

        # Contar cruces por cero para cada columna
        cross_zero_counts = {}
        for col in boc_normalized.columns:
            cross_zero_counts[col] = ((boc_normalized[col].shift(1) * boc_normalized[col]) < 0).sum()

        # Convertir el resultado en un DataFrame
        df = pd.DataFrame(list(cross_zero_counts.items()), columns=['Column', 'Zero_Crossings'])
        df = df.set_index('Column')

        # Ordenar los datos en orden descendente por 'Zero_Crossings'
        df_sorted = df.sort_values(by='Zero_Crossings', ascending=False).reset_index()

        # Calcular el porcentaje acumulativo
        df_sorted['Cumulative_Percentage'] = df_sorted['Zero_Crossings'].cumsum() / df_sorted['Zero_Crossings'].sum() * 100

        # Crear el gráfico de área
        plt.figure(figsize=(10, 6))
        plt.fill_between(
            df_sorted['Column'],
            df_sorted['Zero_Crossings'],
            color="skyblue",
            alpha=0.4,
            label="Zero Crossings"
        )

        # Configuración de ejes y leyendas
        plt.xlabel('Column')
        plt.ylabel('Zero Crossings')
        plt.title('Area Chart of Zero Crossings')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        # Mostrar el gráfico
        figu=plt.show()

        return figu
    
    def cejas(dfg):
        cejas=dfg[['F1EBX','F2EBS','F3EBS','F4EBX']]
                
        # Estandarización de las columnas numéricas
        numeric_cols = cejas.select_dtypes(include=['number']).columns
        scaler = StandardScaler()
        boc_normalized = pd.DataFrame(scaler.fit_transform(cejas[numeric_cols]), columns=numeric_cols)

        # Contar cruces por cero para cada columna
        cross_zero_counts = {}
        for col in boc_normalized.columns:
            cross_zero_counts[col] = ((boc_normalized[col].shift(1) * boc_normalized[col]) < 0).sum()

        # Convertir el resultado en un DataFrame
        df = pd.DataFrame(list(cross_zero_counts.items()), columns=['Column', 'Zero_Crossings'])
        df = df.set_index('Column')

        # Ordenar los datos en orden descendente por 'Zero_Crossings'
        df_sorted = df.sort_values(by='Zero_Crossings', ascending=False).reset_index()

        # Calcular el porcentaje acumulativo
        df_sorted['Cumulative_Percentage'] = df_sorted['Zero_Crossings'].cumsum() / df_sorted['Zero_Crossings'].sum() * 100

        # Crear el gráfico de área
        plt.figure(figsize=(10, 6))
        plt.fill_between(
            df_sorted['Column'],
            df_sorted['Zero_Crossings'],
            color="skyblue",
            alpha=0.4,
            label="Zero Crossings"
        )

        # Configuración de ejes y leyendas
        plt.xlabel('Column')
        plt.ylabel('Zero Crossings')
        plt.title('Area Chart of Zero Crossings')
        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()

        # Mostrar el gráfico
        figu=plt.show()
        return figu
    
    col9,col10=st.columns(2)
    with col9:
        col11,col12=st.columns(2)
        with col11:
            st.markdown('Gráfica Sesión 1')
            a,b= pir(df1)
            st.pyplot(a)
           # Renombrar el índice 0 a "Sumatoria Total"
            # Mostrar el DataFrame en Streamlit
            st.write('Sumatorias totales de los puntos de la cara')
            # Renombrar el índice 0 a "Sumatoria Total" si es necesaria
            # Resetear el índice para que no se muestre en la tabla
            b_reset = b[['Derecho', 'Izquierdo']]

            # Mostrar el DataFrame sin el índice en Streamlit
            st.table(b_reset)


            
            st.markdown('Gráfica Sesión 3')
            a,b= pir(df3)
            st.pyplot(a)
            # Mostrar el DataFrame en Streamlit
            st.write('Sumatorias totales de los puntos de la cara')


            # Resetear el índice para que no se muestre en la tabla
            b_reset = b[['Derecho', 'Izquierdo']].reset_index(drop=True)

            # Mostrar el DataFrame sin el índice en Streamlit
            st.table(b_reset)
    
            
        with col12:
            st.markdown('Gráfica Sesión 2')
            a,b= pir(df2)
            st.pyplot(a)
            # Mostrar el DataFrame en Streamlit
            st.write('Sumatorias totales de los puntos de la cara')
            st.table(b[['Derecho', 'Izquierdo']])

            st.markdown('Gráfica Sesión 4')
            a,b= pir(df4)
            st.pyplot(a)
            # Mostrar el DataFrame en Streamlit
            st.write('Sumatorias totales de los puntos de la cara')
            st.table(b[['Derecho', 'Izquierdo']])
            

    # Agregar la imagen a la columna col10
    with col10:
        st.image(
            "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTP619bV1ShBa6yeDizW8inPj9NJQ_4_g12nETuo33JRlhCfaTysc8WKXLXWl-WgKLkE2w&usqp=CAU", 
            caption="Puntos de estudio en figura de la cara", 
            use_container_width=True
        )
        # Línea divisora
        st.write('Los gráficas de color azul hacen referencia a los puntos del lado derecho del rostro, por otro lado el lado izquierdo está representado de color rosa')
        
        col15, col16 = st.columns(2)
        with col15:
            st.markdown('Gráfica Sesiones En Boca')
            a= boca_multiple_sesions(df1)
            b= boca_multiple_sesions(df3)
            c=boca_multiple_sesions(df2)
            d= boca_multiple_sesions(df3)
            st.pyplot(a)
            st.pyplot(b)
            st.pyplot(c)
            st.pyplot(d)


    
            
        with col16:
            st.markdown('Gráfica Sesión 2')
            a= boca_multiple_sesions(df2)
            b= boca_multiple_sesions(df4)
            st.pyplot(a)
           # Renombrar el índice 0 a "Sumatoria Total"
            # Mostrar el DataFrame en Streamlit
            
            st.markdown('Gráfica Sesión 4')
            st.pyplot(b)
            # Mostrar el DataFrame en Streamlit
            st.write('Gráfica cruces en 0 BOCA')
