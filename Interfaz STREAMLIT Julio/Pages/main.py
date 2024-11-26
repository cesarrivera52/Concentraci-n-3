import streamlit as st 

def create_sidebar():
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "main.py"

    with st.sidebar:
        st.page_link("main.py", label="Inicio", icon=":material/home:")
        st.subheader("Tableros")
        st.page_link("pages/page1.py", label="Estadísticas", icon=":material/home:")
        st.page_link("pages/page2.py", label="Predicciones", icon=":material/shopping_cart:")
        st.page_link("pages/page3.py", label="Comparativo", icon=":material/group:")

    return st.session_state.selected_page 

selected_page = create_sidebar()

st.title("Página principal") 
