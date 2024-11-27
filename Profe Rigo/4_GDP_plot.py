from dash import Dash, html, dcc, callback, Output, Input 
import plotly.express as px 
import pandas as pd 

#Creación de app de dash 
app= Dash(__name__)

#carga de datos
df = pd.read_csv(r'gdp-life-exp-2007.csv') 

fig = px.scatter(df, x= 'gdp per capita', y='life expectancy',
                 size='population', color='continent', hover_name='country',
                 log_x=True, size_max=60)

app.layout = html.Div([
    dcc.Graph(
        id='life-exp vs gdp',
        figure= fig
    )
])

if __name__ =='__main__':
    app.run(debug=True) 