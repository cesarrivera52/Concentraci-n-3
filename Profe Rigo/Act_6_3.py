#Actividad 6.3 Interactividad en Dash
#▹Menú Dropdown para seleccionar valor a la apertura o al cierre (aplicando interactividad, afecta al gráfico 1 de línea).

# ▹DatePickerRange para indicar rango de fecha afectando a los 2 gráficos siguientes:
# ▹Gráfico 1 de línea con el valor de cotización del índice histórico (Columna Close por defecto)
# ▹Gráfico de barras con el volumen negociado de manera histórica (Columna Volume)

# Importación de librerías
from dash import Dash, html, dcc
import plotly.graph_objs as go
import pandas as pd
from dash.dependencies import Input, Output
from dash.dcc import DatePickerRange

# Creación de la app de dash
app = Dash(__name__)

# Carga datos
df_sp500 = pd.read_csv(r'SP500_data_.csv', encoding='ISO-8859-1', delimiter=',')

# Objetos plotly.graph
data1 = [go.Scatter(x=df_sp500['Date'],
                    y=df_sp500['Close'],
                    mode='markers',
                    marker=dict(
                        size=12,
                        symbol='circle',
                        line={'width': 3}
                    ))]

layout1 = go.Layout(title='Cierre SP500',
                    xaxis=dict(title='fecha'),
                    yaxis=dict(title='precio de cierre'))

data2 = [go.Bar(x=df_sp500['Date'],
                y=df_sp500['Volume']
                )]

layout2 = go.Layout(title='Volumen negociado de manera histórica',
                    xaxis=dict(title='fecha'),
                    yaxis=dict(title='volumen histórico'),
                    font=dict(color='darkmagenta'),
                    paper_bgcolor="MediumTurquoise")

app.layout = html.Div([
    html.Label('Selecciona apertura o cierre'),
    dcc.Dropdown(
        options=[
            {'label': 'Apertura', 'value': 'Open'},
            {'label': 'Cierre', 'value': 'Close'}
        ],
        value='Open',
        id='dropdown-apertura-cierre'
    ),
    html.Label('Selecciona el rango de fechas'),
    DatePickerRange(
        id='date-picker-range',
        start_date=df_sp500['Date'].min(),
        end_date=df_sp500['Date'].max(),
        display_format='YYYY-MM-DD'
    ),
    dcc.Graph(
        id='scatter'
    ),
    dcc.Graph(
        id='barplot'
    )
])

@app.callback(
    [Output('scatter', 'figure'),
     Output('barplot', 'figure')],
    [Input('dropdown-apertura-cierre', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graphs(selected_value, start_date, end_date):
    filtered_df = df_sp500[(df_sp500['Date'] >= start_date) & (df_sp500['Date'] <= end_date)]
    
    scatter_data = [go.Scatter(x=filtered_df['Date'],
                               y=filtered_df[selected_value],
                               mode='markers',
                               marker=dict(
                                   size=7,
                                   symbol='cross-open',
                                   line={'width': 3}
                               ))]
    scatter_layout = go.Layout(title=f'{selected_value} SP500',
                               xaxis=dict(title='Fecha'),
                               yaxis=dict(title=f'Precio de {selected_value.lower()}'),
                               font=dict(color='navy'),  # Color de la fuente
                               paper_bgcolor="rgb(255, 160, 122)",  # Color de fondo del gráfico
                               plot_bgcolor="rgb(216, 191, 216)") 
    
    bar_data = [go.Bar(x=filtered_df['Date'],
                   y=filtered_df['Volume'],
                   marker=dict(color='purple'))]  # Color de las barras

    bar_layout = go.Layout(title='Volumen negociado de manera histórica',
                       xaxis=dict(title='Fecha', titlefont=dict(color='darkgreen')),  # Color del título del eje x
                       yaxis=dict(title='Volumen histórico', titlefont=dict(color='darkgreen')),  # Color del título del eje y
                       font=dict(color='navy'),  # Color de la fuente
                       paper_bgcolor="lightcoral",  # Color de fondo del gráfico
                       plot_bgcolor="lightcyan")  # Color de fondo del área del gráfico
    
    return {'data': scatter_data, 'layout': scatter_layout}, {'data': bar_data, 'layout': bar_layout}

if __name__ == '__main__':
    app.run(debug=True)