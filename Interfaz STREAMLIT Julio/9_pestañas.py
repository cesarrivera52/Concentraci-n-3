import streamlit as st

st.title("Ejemplo de pestañas")

#tab1, tab2, tab3 = st.tabs(["Datos de ventas", "Análisis de clientes", "Gráficos"])
tab1, tab2, tab3 = st.tabs([":bookmark_tabs: Datos de ventas", "$\mathbf{Análisis de clientes}$", ":chart_with_upwards_trend: :blue[Gráficos]"])

with tab1:
    st.header("Datos de ventas")
    st.write("Aquí se muestran los datos de ventas recientes")
    st.dataframe({"Producto": ["A","B","C"], "Ventas": [100, 150, 80]})

with tab2:
    st.header("Análisis de clientes")
    st.write("Información detallada sobre clientes")
    st.text_area("Observaciones", "Aquí puedes escribir tus observaciones")

with tab3:
    st.header("Gráficos")
    st.line_chart([3,5,6,7,1,8,2,7])



