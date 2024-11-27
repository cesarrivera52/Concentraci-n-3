from dash import Dash, html, dcc, callback, Output, Input 

#Creación de app de dash 
app= Dash(__name__)

#Definición de diccionario de los colores utilizados en la app
colors ={
    'background': '#E82862',
    'text': '#7FDBFF'
}

#Cargar el layout 

app.layout = html.Div([ #invocar el objeto Div 
    
    html.H1(children= "Primer dashboard con Dash", #Objetos de cabecera, children= Propiedad base de cada uno de los elementos
            style={ #definición del estilo del componente H1
                'textAlign': 'center',
                'color': colors['text']
            }),
    html.Div(children= "Dash es un framework para aplicaciones web", #crear otro contenedor
            style={ #definición del estilo del componente H1
                'textAlign': 'center',
                'color': colors['text']
            }),
    dcc.Graph(id= "Figura inicial",
              figure={
                  "data":[
                      {"x":[1,2,3], "y":[5,9,4], "type":"bar", "name":"Puebla"},
                       {"x":[3,5,1], "y":[6,8,7], "type":"bar", "name":"Queretaro"}
                  ],
                  "layout":{"title":"Comparativa Puebla-Queretaro",
                            'plot_bgcolor': colors['background'],
                            'paper_bgcolor': colors['background'],
                            'font':{'color':colors['text']}
                            }
              }
                           
              )
]
) 

if __name__ =='__main__':
    app.run(debug=True) 