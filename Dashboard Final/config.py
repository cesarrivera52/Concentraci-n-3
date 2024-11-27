import streamlit as st
#@st.cache_data
import os
import pandas as pd
def load_df(data):
    import pandas as pd
    import datetime
    import math
    data.dropna(inplace=True)
    def distance(x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)
    data['Distancia'] = data.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
    data['01_C']= data['01_C'].replace({-1:0})
    data['02_A']= data['02_A'].replace({-1:0})
    data['03_D']= data['03_D'].replace({-1:0})
    data['04_M']= data['04_M'].replace({-1:0})
    return data

folder_paths = ['./Paciente 1','./Paciente 2','./Paciente 3']
# Define la ruta de la carpeta
folder_path = './Paciente 1'
@st.cache_data
def read_patient(folder_path):
    # List the files in the folder
    files = os.listdir(folder_path)

    # Initialize an empty list to store the DataFrames
    dataframes = []

    # Read all CSV files and store their content in the list
    for file_name in files:
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            df = pd.read_csv(file_path)
            dataframes.append(df)
    return dataframes
datas=[]
for path in folder_paths:
    datas.append(read_patient(path))
df1p1=load_df(datas[0][0])
df2p1=load_df(datas[0][1])
df3p1=load_df(datas[0][2])
df4p1=load_df(datas[0][3])

df1p2=load_df(datas[1][0])
df2p2=load_df(datas[1][1])
df3p2=load_df(datas[1][2])
df4p2=load_df(datas[1][3])

df1p3=load_df(datas[2][0])
df2p3=load_df(datas[2][1])
df3p3=load_df(datas[2][2])
df4p3=load_df(datas[2][3])