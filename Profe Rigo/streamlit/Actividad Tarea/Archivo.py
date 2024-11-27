import streamlit as st
import pandas as pd
import plotly.express as px

############ César Alejandro Rivera Guzmán A01567012


st.title('Actividad 6.4 Despliegue de gráficas con Streamlit')
st.write('Esta actividad tiene como objetivo poder mostrar datos del socioformador, como gráficas e información relevante')

# Carga del archivo
archivo = st.sidebar.file_uploader('Carga el archivo CSV', type=['csv'])

if archivo:
    # Cargar los datos
    datos = pd.read_csv(archivo, encoding='iso-8859-1')
    datos['tiempo'] = datos.index * 3  # Crea una columna de tiempo basada en el índice

    columnas = datos.columns  # Obtén las columnas del DataFrame
    data=datos.copy()

    ####################### Sidebar #######################
    # Mostrar el DataFrame si el botón está activado
    boton = st.button(label='Mostrar Data Frame')
    if boton:
        st.write('Vista previa de los datos cargados:')
        st.dataframe(datos.head())
    
    # Selección de columnas
    st.sidebar.header('1. Punto Uno')
    colum = st.sidebar.multiselect(
        f'Selecciona la(s) **columna(s)** que deseas comparar vs tiempo', 
        options=columnas
    )

    # Construcción dinámica del texto del botón
    if colum:
        colum_joined = ", ".join(colum)  # Combina las columnas seleccionadas en una sola cadena
        boton_label = f'Punto 1'
    else:
        boton_label = 'Punto 1'

    # Renderiza el botón con el nombre dinámico
    if st.checkbox(boton_label):
        st.header('1. Punto Uno')
        st.write(f"Has seleccionado las columnas: {colum}")

        # Prepara los datos para el gráfico
        datos_melted = datos.melt(
            id_vars='tiempo', 
            value_vars=colum, 
            var_name='Columna', 
            value_name='Valor'
        )

        # Crear gráfico de barras múltiples con Plotly
        fig = px.bar(
            datos_melted, 
            x='tiempo', 
            y='Valor', 
            color='Columna', 
            barmode='group',  # Opciones: 'group' para barras agrupadas, 'stack' para apiladas
            title='Multi-Bar Chart - Comparación de columnas vs Tiempo'
        )

        # Mostrar el gráfico
        st.plotly_chart(fig)

    import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# Segunda parte: Punto Dos
st.sidebar.header('2. Punto Dos')
variab = st.sidebar.selectbox(
    'De las siguientes opciones, se desplegarán 4 gráficas para poder observar su comportamiento', 
    ['Veloc', 'Acele', 'Des_x', 'Des_y']
)

if variab:
    boton2 = st.checkbox('Punto 2')
    if boton2:
        st.title('2. Punto 2')
        with st.container():
            st.subheader("Se desplegarán 4 gráficas para poder observar su comportamiento")
            st.text("Este contenedor presenta información adicional sobre el comportamiento de las variables seleccionadas")

            # Selector de tipo de gráfico
            chart_type = st.selectbox(
                'Seleccionar tipo de gráfico',
                ['Barras', 'Histograma', 'Densidad', 'Alejada']
            )

            # Gráfico de Barras
            if chart_type == 'Barras':
                st.write(f"Gráfico de barras para la variable: {variab}")

                # Crear la gráfica de barras usando Matplotlib
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(datos['tiempo'], datos[variab], color='skyblue', alpha=0.8)

                # Configuración del gráfico
                ax.set_title(f'Gráfico de Barras - {variab}', fontsize=14)
                ax.set_xlabel('Tiempo', fontsize=12)
                ax.set_ylabel('Valor', fontsize=12)
                plt.xticks(rotation=45)
                
                # Mostrar el gráfico
                st.pyplot(fig)

            # Histograma
            elif chart_type == 'Histograma':
                st.write(f"Histograma para la variable: {variab}")

                # Crear el histograma usando Plotly
                fig = px.histogram(
                    datos,
                    x=variab,
                    nbins=20,
                    title=f'Histograma - {variab}',
                    labels={variab: 'Valor', 'count': 'Frecuencia'}
                )
                st.plotly_chart(fig)

            # Densidad
            elif chart_type == 'Densidad':
                st.write(f"Gráfico de densidad para la variable: {variab}")

                # Crear gráfico de densidad con Plotly
                fig = px.density_contour(
                    datos,
                    x='tiempo',
                    y=variab,
                    title=f'Densidad - {variab}',
                    labels={'x': 'Tiempo', 'y': variab}
                )
                st.plotly_chart(fig)

            # Gráfico Alejado
            elif chart_type == 'Alejada':
                st.write(f"Gráfico alejado (Scatter Plot) para la variable: {variab}")

                # Crear Scatter Plot con Plotly
                fig = px.scatter(
                    datos,
                    x='tiempo',
                    y=variab,
                    title=f'Scatter Plot - {variab}',
                    labels={'x': 'Tiempo', 'y': variab}
                )
                st.plotly_chart(fig)

st.sidebar.header('3. Punto Tres')
st.sidebar.markdown('Gráfica de la suma de cada columna.')

# Checkbox para activar este punto
boton3 = st.checkbox('Punto 3')

if boton3:
    st.subheader('Suma Total por Columna')

    # Excluir la última columna
    columnas_numericas = datos.select_dtypes(include=['int64', 'float64']).columns[:-1]  # Excluye la última columna

    # Crear DataFrame con las sumas
    suma_columnas = pd.DataFrame({
        'Columna': columnas_numericas,
        'Suma Total': [datos[col].sum() for col in columnas_numericas]
    })

    # Mostrar el DataFrame con las sumas
    st.write("Suma de cada columna:")
    st.dataframe(suma_columnas)

    # Crear Multi-Bar Chart
    fig = px.bar(
        suma_columnas,
        x='Columna',
        y='Suma Total',
        title='Suma Total de Cada Columna',
        labels={'Suma Total': 'Suma Total', 'Columna': 'Columnas'},
        color='Columna'  # Colorear las barras por columna
    )

    # Mostrar el gráfico
    st.plotly_chart(fig)

# Configuración del sidebar para Punto 4
st.sidebar.header('4. Punto Cuatro')
st.sidebar.markdown('Gráfica de presencia de sentimientos en intervalos de tiempo.')

# Checkbox para activar este punto
boton4 = st.checkbox('Punto 4')

if boton4:
    st.header('Punto Cuatro')
    st.subheader('Presencia de Sentimientos en Intervalos de Tiempo')

    # Convertir la columna de tiempo a minutos
    data['tiempo_min'] = data['tiempo'] / 60  # Suponiendo que 'tiempo' está en segundos

    # Configurar slider para seleccionar rango de tiempo
    tiempo_min = data['tiempo_min'].min()
    tiempo_max = data['tiempo_min'].max()

    rango_tiempo = st.slider(
        'Selecciona el rango de tiempo en minutos:',
        min_value=tiempo_min,
        max_value=tiempo_max,
        value=(tiempo_min, tiempo_max),  # Rango por defecto
        step=1.0
    )

    # Filtrar los datos según el rango seleccionado
    data_filtrada = data[(data['tiempo_min'] >= rango_tiempo[0]) & (data['tiempo_min'] <= rango_tiempo[1])]

    # Columnas de sentimiento
    sentiment_columns = ['01_C', '02_A', '03_D', '04_M']

    # Crear la figura con subplots (uno por sentimiento)
    fig, axs = plt.subplots(nrows=len(sentiment_columns), ncols=1, figsize=(10, 12))

    # Iterar sobre cada columna de sentimiento y su subplot correspondiente
    for i, (col, ax) in enumerate(zip(sentiment_columns, axs)):
        # Filtro los intervalos de tiempo donde el sentimiento está presente (valor = 1)
        intervalos = []
        start = None

        for j in range(len(data_filtrada)):
            if data_filtrada[col].iloc[j] == 1:
                if start is None:
                    start = data_filtrada['tiempo_min'].iloc[j]
            else:
                if start is not None:
                    end = data_filtrada['tiempo_min'].iloc[j - 1]
                    intervalos.append((start, end))
                    start = None

        # Capturar el último intervalo si termina en el último valor de la columna
        if start is not None:
            intervalos.append((start, data_filtrada['tiempo_min'].iloc[-1]))

        # Dibujar barras horizontales para los intervalos
        for (start, end) in intervalos:
            ax.hlines(y=0, xmin=start, xmax=end, color=plt.get_cmap("Set2")(i), linewidth=6)

        # Etiquetas y formato
        ax.set_yticks([0])
        ax.set_yticklabels([col])
        ax.set_xlabel("Tiempo (minutos)")
        ax.set_xlim(rango_tiempo[0], rango_tiempo[1])
        ax.set_title(f"Presencia de {col} en intervalos de tiempo")
        ax.grid(axis='x', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Mostrar el gráfico en Streamlit
    st.pyplot(fig)
