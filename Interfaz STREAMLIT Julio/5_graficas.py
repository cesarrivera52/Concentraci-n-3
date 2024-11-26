import numpy as np 
import pandas as pd 
import streamlit as st 
import matplotlib.pyplot as plt 

st.title("Gráfica de datos aleatorios")
chart_data = pd.DataFrame(
    np.random.randn(20,3),
    columns = ['a','b','c']
)

st.line_chart(chart_data) 

st.write("Distribución en forma de campana para los lanzamientos de una moneda")

perc_heads = st.number_input(label='Probabilidad de éxito para que las monedas caigan en cara',
                             min_value=0.0, max_value=1.0, value=0.5)

graph_title = st.text_input(label='Titulo de la gráfica') 

binom_dist = np.random.binomial(1, perc_heads, 1000) 
#binom_dist = np.random.binomial(1, .5, 1000) 
list_of_means = []
for i in range(0,1000):
    list_of_means.append(np.random.choice(binom_dist, 100, replace=True).mean())
fig, ax = plt.subplots()
ax = plt.hist(list_of_means)
plt.title(graph_title) 

st.pyplot(fig) 

####Creación de un mapa 
map_data = pd.DataFrame(
    np.random.randn(50,2) / [50,50] + [19.04, -98.2],
    columns=['lat','lon'])

st.map(map_data) 


