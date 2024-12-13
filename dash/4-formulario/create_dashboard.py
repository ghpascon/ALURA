from ucimlrepo import fetch_ucirepo

def dash_page(model):
    #imports
    import os
    from dash import Dash, dcc, html
    import dash_bootstrap_components as dbc

    from dash_components import form_components, callbacks

    #pagina
    app = Dash(
        __name__,
        external_stylesheets=["dash_components/css.css",dbc.themes.FLATLY],
        suppress_callback_exceptions=True
    )

    navbar = dbc.NavbarSimple(
        brand="Dashboard",
        brand_href="#",
        color="primary",
        dark=True,
        children=[
            dbc.DropdownMenu(
                in_navbar=True,
                label="Menu",
                children=[
                    dbc.DropdownMenuItem("Gráficos", href='/graficos'),
                    dbc.DropdownMenuItem("Formulário", href='/formulario'),
                ],
            ),
        ],
    )

    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        navbar,
        html.Div(id='page-content')
    ])
    
    callbacks.dash_callbacks(app, model)

# Código omitido

    app.run_server(debug=False, port=8080, host='0.0.0.0')        

def load_model(model_dir_name, filename):
    import pickle
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, model_dir_name)
    nome_arquivo = os.path.join(model_dir, f"{filename}.pkl")
    try:
        with open(nome_arquivo, 'rb') as arquivo:
            model = pickle.load(arquivo)
            print("Modelo carregado com sucesso!")
            return model

    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")   
        exit(1)

if __name__ == "__main__":
    model_dir="models"
    filename="model_xgboost"
    model = load_model(model_dir, filename)

    dash_page(model)

