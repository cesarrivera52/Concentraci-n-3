import pandas as pd 
import streamlit as st 
import plotly.express as px 

st.set_page_config(page_title='Árboles San Fransisco', layout='wide')

st.title("SF Trees")
st.write("Esta aplicación analiza árboles de San Fransisco") 

trees_df = pd.read_csv("trees.csv") 
st.write(trees_df.head()) 

today = pd.to_datetime("today")
trees_df["date"] = pd.to_datetime(trees_df["date"])
trees_df["age"] = (today - trees_df["date"]).dt.days
unique_caretakers = trees_df["caretaker"].unique()
owners = st.sidebar.multiselect("Tree owner filter", unique_caretakers)
graph_color = st.sidebar.color_picker("Graph colors")
if owners:
    trees_df = trees_df[trees_df["caretaker"].isin(owners)]


# Crear columnas y gráficos
col1, col2 = st.columns(2, gap='large')

with col1:
    fig = px.histogram(
        trees_df,
        x=trees_df["dbh"],
        title="Tree Width",
        color_discrete_sequence=[graph_color]
    )
    fig.update_xaxes(title_text="Width")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        trees_df,
        x=trees_df["age"],
        title="Tree Age",
        color_discrete_sequence=[graph_color]
    )

    st.plotly_chart(fig, use_container_width=True)






