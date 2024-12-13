from dash.dependencies import Input, Output, State
from sklearn.metrics import accuracy_score
import pandas as pd
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

import os

import importlib.util
script_dir = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.join(script_dir, "pages.py")

spec = importlib.util.spec_from_file_location("pages", module_path)
pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pages)


def dash_callbacks(app, model):
    @app.callback(
        Output('predict', 'children'),
        Input('submit-button', 'n_clicks'),
        [
            State('idade', 'value'),
            State('sexo', 'value'),
            State('cp', 'value'),
            State('trestbps', 'value'),
            State('chol', 'value'),
            State('fbs', 'value'),
            State('restecg', 'value'),
            State('thalach', 'value'),
            State('exang', 'value'),
            State('oldpeak', 'value'),
            State('slope', 'value'),
            State('ca', 'value'),
            State('thal', 'value'),
        ],
        prevent_initial_call=True,
    )
    def prever_doenca(n_clicks, *args):
        try:
            # Lista de nomes das variáveis
            columns = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
            ]
            
            # Inicializar o dicionário de dados
            data_dict = {}
            
            # Iterar sobre os valores e processar
            for col, value in zip(columns, args):
                if value is None:
                    return dbc.Alert("Preencha todos os Campos", color="warning") 
                    
                if col == 'oldpeak':  # Caso específico para valores float
                    data_dict[col] = float(value) if value is not None else 0.0
                else:  # Todos os outros como inteiros
                    data_dict[col] = int(value) if value is not None else 0
            
            # Criar o DataFrame
            data = pd.DataFrame([data_dict])
            
            # Fazer a previsão
            predict = model.predict(data)[0]
            if predict == 1:
                return dbc.Alert("Voce tem doença cardiaca", color="danger") 
            else:
                return dbc.Alert("Voce não tem doença cardiaca", color="success") 
                
            # Exibir o resultado

        except Exception as e:
            return f"Erro ao processar a previsão: {e}"


    @app.callback(
        Output('page-content', 'children'),
        Input('url', 'pathname'),
    )
    def mostrar_pagina(pathname):
        if pathname == '/formulario':
            return pages.formulario
        if pathname == '/graficos':
            return pages.graficos

        return 'home'
    
