import streamlit as st 

st.title("Despliegue de columnas")

first_width = st.number_input('First Width', min_value =1, value=1)
second_width = st.number_input('Second Width', min_value =1, value=1)
third_width = st.number_input('Third Width', min_value =1, value=1)

col1, col2, col3 = st.columns([first_width, second_width, third_width])
with col1:
    st.write("First Column")
with col2:
    st.write("Second Column")
with col3:
    st.write("Third Column") 


col1, col2, col3 = st.columns(3)
with col1:
    st.write("First Column")
with col2:
    st.write("Second Column")
with col3:
    st.write("Third Column") 


col1, col2, col3 = st.columns([1,2,3])
with col1:
    st.write("First Column")
with col2:
    st.write("Second Column")
with col3:
    st.write("Third Column") 


col1, col2 = st.columns([0.7,0.3])
with col1:
    st.write("First Column")
with col2:
    st.write("Second Column")