import pandas as pd 
import streamlit as st 

st.set_page_config(page_title='Árboles San Fransisco', layout='wide')

st.title("SF Trees")
st.write("Esta aplicación analiza árboles de San Fransisco") 

trees_df = pd.read_csv("trees.csv") 
st.write(trees_df.head()) 

df_dhb_grouped = pd.DataFrame(trees_df.groupby(["dbh"]).count()["tree_id"]) 
df_dhb_grouped.columns = ["tree_count"]

# Crear columnas y gráficos
col1, col2, col3 = st.columns(3)
with col1:
    st.line_chart(df_dhb_grouped)
with col2:
    st.bar_chart(df_dhb_grouped)
with col3:
    st.area_chart(df_dhb_grouped)

trees_df = trees_df.dropna(subset=["longitude","latitude"])
trees_df = trees_df.sample(n=1000)
st.map(trees_df)

