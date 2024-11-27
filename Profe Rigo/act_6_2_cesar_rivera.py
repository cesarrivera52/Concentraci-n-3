# Importación de librerías
from dash import Dash, html, dcc, Input, Output
import plotly.graph_objs as go
import pandas as pd

# Creación de la app de dash
app = Dash(__name__)

# Carga de datos
df = pd.read_csv(r'SP500_data_.csv', encoding='ISO-8859-1', delimiter=',')
df['Date'] = pd.to_datetime(df['Date'])  # Asegúrate de que la fecha esté en formato datetime

# Objetos plotly.graph
data1 = [go.Scatter(x=df['Date'], #eje x
                    y=df['Close'], # eje y
                    mode='markers', 
                    marker=dict(
                        size=10,
                        symbol='bowtie', #tipo de simbolo que aparecerá en la gráfica
                        line={'width': 2}
                    ))]

layout1 = go.Layout(title='valor de cierre de la cotización del índice histórico',
                    xaxis=dict(title='Fecha'),
                    yaxis=dict(title='Precio cierre'),
                    font=dict(color='darkmagenta'), # color de los strings
                    paper_bgcolor="#FFABAB") # color del background 

data2 = [go.Bar(x=df['Date'],
                 y=df['Volume'])]

layout2 = go.Layout(title='volumen negociado de manera histórica',
                    xaxis=dict(title='Fecha'),
                    yaxis=dict(title='Volumen'),
                    font=dict(color='darkmagenta'), # color de los strings
                    paper_bgcolor="#A2C2E0")  # color del background

app.layout = html.Div([
    # Dropdown para seleccionar el usuario
    html.Label('Selecciona apertura o cierre'),
    dcc.Dropdown(
        options=[
            {'label': 'Apertura', 'value': 'Open'},
            {'label': 'Cierre', 'value': 'Close'}
        ],
        value='Close',  # Valor seleccionado por defecto
        id='dropdown'  # Añadir un ID para el callback
    ),
    dcc.Graph(id='scatterplot',
              figure={'data': data1,
                      'layout': layout1}),
    dcc.Graph(id='barplot',
              figure={'data': data2,
                      'layout': layout2})
])

# Callback para actualizar el gráfico de dispersión basado en la selección del dropdown
@app.callback(
    Output('scatterplot', 'figure'),
    Input('dropdown', 'value')
)
def update_graph(selected_value):
    if selected_value == 'Open':
        y_data = df['Open']
    else:
        y_data = df['Close']

    # Actualizar el gráfico de dispersión
    updated_data = [go.Scatter(x=df['Date'],
                                y=y_data,
                                mode='markers',
                                marker=dict(
                                    size=12,
                                    symbol='bowtie',
                                    line={'width': 2}
                                ))]

    return {'data': updated_data, 'layout': layout1}

if __name__ == '__main__':
    app.run(debug=True)

