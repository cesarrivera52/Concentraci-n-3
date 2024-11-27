from dash import Dash, html, dcc, callback, Output, Input 
import plotly.graph_objects as go 
import pandas as pd 

#Creación de app de dash 
app= Dash(__name__)

#carga de datos
df_iris = pd.read_csv(r'iris_dataset.csv', encoding='ISO-8859-1', delimiter=',') 

#objetos plotly.graph

data1 = [go.Scatter(x=df_iris['longitud_sépalo'],
                    y=df_iris['anchura_sépalo'],
                    mode='markers',
                    marker = dict(
                        size=12,
                        symbol='circle',
                        line={'width':3}
                    ))] 

layout1 = go.Layout(title='Iris Scatter Plot Sépalo',
                    xaxis=dict(title='Longitud Sépalo'),
                    yaxis=dict(title='Anchura Sépalo')) 

data2 = [go.Scatter(x=df_iris['longitud_pétalo'],
                    y=df_iris['anchura_pétalo'],
                    mode='markers',
                    marker = dict(
                        size=12,
                        symbol='hexagon-dot',
                        line={'width':3},
                        color='cyan'
                    ))] 

layout2 = go.Layout(title='Iris Scatter Plot Pétalo',
                    xaxis=dict(title='Longitud pétalo'),
                    yaxis=dict(title='Anchura pétalo'),
                    font= dict(color='darkmagenta'),
                    paper_bgcolor='MediumTurquoise') 

app.layout = html.Div([
    dcc.Graph(id='scatterplot_1',
              figure={'data':data1,
                      'layout': layout1}), 

    dcc.Graph(id='scatterplot_2',
              figure={'data':data2,
                      'layout': layout2}) 

]) 

if __name__ =='__main__':
    app.run(debug=True) 