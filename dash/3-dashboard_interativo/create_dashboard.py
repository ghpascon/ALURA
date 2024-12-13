from ucimlrepo import fetch_ucirepo
import plotly.express as px
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score 

def get_dataframe():
    heart_disease = fetch_ucirepo(id=45)
    
    # Data (as pandas DataFrames)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Variable information
    print(heart_disease.variables)

    return X, y

def get_model_data(data, y_label):
    if data.empty:
        print("Os dados estão vazios.")
        return None,None
    
    x = data.drop(columns=[y_label])
    y = data[y_label]
    
    return x, y

def dash_page():
    from dash import Dash, dcc, html

    app = Dash(__name__)

    app.layout = html.Div([
        html.Div([
            html.Label("Coloque sua idade:  "),
            dcc.Input(id="idade", type="number", value=0),
        ]),
        html.Div(id="meses_result"),

        html.Div([
            html.Label("Coloque seu Nome:  "),
            dcc.Input(id="name", type="text", value=""),
            html.Button("Submit", id="submit_bt", n_clicks=0),
        ]),
        html.Div(id="name_result"),
    ])

    dash_callbacks(app)

    app.run_server(debug=True)

def dash_callbacks(app):
    from dash.dependencies import Input, Output, State
    #callback idade
    @app.callback(
        output = Output('meses_result', 'children'),
        inputs = Input('idade', 'value'),
        prevent_initial_call=True
    )
    def calcula_meses(idade):
        if idade is not None:
            return f"Sua idade em meses é: {idade * 12}"
        return "Por favor, insira uma idade válida." 

    @app.callback(
        output = Output('name_result', 'children'),
        inputs = Input('submit_bt', 'n_clicks'),
        state = State('name', 'value'),
        prevent_initial_call=True
    )
    def retorna_nome(n_clicks, name):
        if name is not None:
            return f"Seu nome é: {name}"
        return "Por favor, insira um Nome válido."           

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
    # model = load_model(model_dir, filename)

    dash_page()