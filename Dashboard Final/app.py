import streamlit as st

create_page = st.Page("PAGE1.py", title="Análisis General", icon="⏱️")
page = st.Page("PAGE2.py", title="Análisis de la cara", icon="🙍‍♂️")

pg = st.navigation([create_page,page])
st.set_page_config(page_title="Tablero Equipo 2", page_icon="🌀",layout="wide")
# Inject custom CSS to set consistent heights and styles for headers  
 
pg.run()