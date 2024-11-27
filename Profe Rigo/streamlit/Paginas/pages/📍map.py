import pandas as pd 
import streamlit as st 
import plotly.express as px 
import pydeck as pdk 

st.set_page_config(page_title="Página 1", layout="wide")

st.markdown(
    """
    <style>
    .stAppHeader{
            background-color: #9932CC;
    }
    .stApp{
        background-color: #00FFFF;
        color: #4CAF50
    }
    .stSidebar{
        background-color: #ffffff;
        color: #333;
        font-family: 'Comic Sans MS';
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("SF Trees Map")
st.write("kTrees by Location") 

trees_df = pd.read_csv("trees.csv") 
trees_df = trees_df.dropna(subset=['longitude','latitude'])
trees_df = trees_df.sample(n= 1000, replace=True)

st.pydeck_chart(
    pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=37.76,
            longitude=-122.4,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
                'HexagonLayer',
                data=trees_df,
                get_position='[longitude, latitude]',
                radius= 200,
                get_color= '[255,75,75]',
                elevation_scale= 10,
                elevation_range=[0,1000],
                pickable= True,
                extruded= True,
            )
        ]
    )
)
