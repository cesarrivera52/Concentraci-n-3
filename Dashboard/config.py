#@st.cache_data
def load_df(arg,ventana):
    import pandas as pd
    import datetime
    import math
    data=pd.read_csv(arg)
    data.dropna(inplace=True)
    data['Tiempo']=data.index*ventana
    data.dropna(inplace=True)
    data['Tiempo']=data['Tiempo'].apply(lambda x: datetime.timedelta(seconds=x).total_seconds() / 60)
    def distance(x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)
    data['Distancia'] = data.apply(lambda row: distance(row['Des_x'], row['Des_y'], row['Des_z']), axis=1)
    data['01_C']= data['01_C'].replace({-1:0})
    data['02_A']= data['02_A'].replace({-1:0})
    data['03_D']= data['03_D'].replace({-1:0})
    data['04_M']= data['04_M'].replace({-1:0})
    return data

df1='11abr.csv'
df2='22abr.csv'
df3='29abr.csv'
df4='03may.csv'
ventana=3
df1=load_df(df1,ventana)
df2=load_df(df2,ventana)
df3=load_df(df3,ventana)
df4=load_df(df4,ventana)