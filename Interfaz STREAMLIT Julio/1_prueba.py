import streamlit as st
import pandas as pd

#st.set_page_config(layout="wide")

st.title('Datos del clima')
st.write('Bienvenidos, en esta app podremos ver el clima a detalle') 


archivo = st.sidebar.file_uploader("Cargar archivo CSV", type=["csv"])
if archivo: 
    datos = pd.read_csv(archivo, encoding='iso-8859-1')
    datos["Fecha"] = pd.to_datetime(datos["Fecha"], format="%d/%m/%Y", errors='coerce')

    #Verificar fechas inválidas 
    if datos["Fecha"].isnull().any():
        st.write("Error: Algunas fechas no se pudieron convertir correctamente")
    else:
        st.write("Vista previa de los datos cargados:")
        st.dataframe(datos.head())

        #Selección del rango de fechas 
        fecha_min = datos["Fecha"].min().date()
        fecha_max= datos["Fecha"].max().date()
        fecha_inicio, fecha_fin = st.sidebar.slider(
            "Selecciona un rango de fechas:",
            min_value = fecha_min,
            max_value = fecha_max,
            value = (fecha_min,fecha_max),
            format = "YYYY-MM-DD"
        )
        #Filtrar datos en base al rango de fechas 
        datos_filtrados = datos[(datos["Fecha"] >= pd.to_datetime(fecha_inicio)) & (datos["Fecha"]<= pd.to_datetime(fecha_fin))]

        columna = st.sidebar.selectbox("Selecciona una columna para ver estadísticas", ("Temperatura","Humedad"))

        #Gráfico de las columna seleccionada 
        st.write(f"Evolución de {columna} entre {fecha_inicio} y {fecha_fin}:")
        st.line_chart(datos_filtrados.set_index("Fecha")[columna])

        #Tabla de estadísticas descriptivas 
        st.write(f"Estadísticas descriptivas de {columna}:")
        st.write(datos_filtrados[columna].describe()[["mean","50%","min","max"]].rename({"50%":"median"}))

        st.write("Gracias por usar la aplicación. Esperamos que la información le haya sido útil")









    

   

