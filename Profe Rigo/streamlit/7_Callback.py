# Importación de librerias
from dash import Dash, html, dcc
from dash.dependencies import Input, Output # Módulos de interactividad

# Creación de la app de dash
app = Dash (__name__)

app.layout = html.Div([
    dcc.Input(id = 'Mi-input', value = 'Valor inicial', type = 'text'),
    html.Div(id='Mi-salida')
])

@app.callback( #Función app decorator
        Output("Mi-salida", "children"),
        [Input("Mi-input", "value")]
)


#Creamos función para interactuar 
def actualizar_div(valor_entrada):
    return 'Has insertado "{}"'.format(valor_entrada)


# Ejecutar la aplicación 
if __name__ == '_main_':
    app.run_server(debug=True)