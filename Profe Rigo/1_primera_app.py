from dash import Dash, html, dcc, callback, Output, Input 

#Creación de app de dash 
app= Dash(__name__)

#Cargar el layout 

app.layout = html.Div([ #invocar el objeto Div 
    
    html.H1(children= "Primer dashboard con Dash"), #Objetos de cabecera, children= Propiedad base de cada uno de los elementos 
    html.Div(children= "Dash es un framework para aplicaciones web"), #crear otro contenedor
    dcc.Graph(id= "Figura inicial",
              figure={
                  "data":[
                      {"x":[1,2,3], "y":[5,9,4], "type":"bar", "name":"Puebla"},
                       {"x":[3,5,1], "y":[6,8,7], "type":"bar", "name":"Queretaro"}
                  ],
                  "layout":{"title":"Comparativa Puebla-Queretaro"}
              }
                           
              )
]
) 

if __name__ =='__main__':
    app.run(debug=True) 