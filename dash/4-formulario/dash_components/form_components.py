from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

title_div = html.Div([
                html.H1("Previsão de Doença do Coração"),
            ],className='text-center mt-5')

age_div = html.Div([
            dbc.Label("Coloque sua idade:  "),
            dbc.Input(id="idade", type="number", placeholder="Digite a Idade"),
        ], className='mb-3')

sex_div = html.Div([
            dbc.Label("Selecione seu sexo biológico:  "),
            dbc.Select(id="sexo", options=[
                {"label": "Masculino", "value": "1"},
                {"label": "Feminino", "value": "0"},
            ]),
        ], className='mb-3')

chest_pain_div = html.Div([
            dbc.Label("Tipo de dor no peito"),
            dbc.Select(id="cp", options=[
                {"label": "Angina típica", "value": "1"},
                {"label": "Angina atípica", "value": "2"},
                {"label": "Não angina", "value": "3"},
                {"label": "Não angina assintomática", "value": "4"},
            ]),
        ], className='mb-3')

trest_bps_div = html.Div([
            dbc.Label("Pressão arterial em repouso:"),
            dbc.Input(id="trestbps", type="number", placeholder="Digite a pressão"),
        ], className='mb-3')


chol_div = html.Div([
            dbc.Label("Colesterol em jejum:"),
            dbc.Input(id="chol", type="number", placeholder="Digite a glicose"),
        ], className='mb-3')

fbs_div = html.Div([
            dbc.Label("Glicose em jejum:"),
            dbc.Select(id="fbs", options=[
                {"label": "Menor uqe 120 mg/dl", "value": "0"},
                {"label": "Maior uqe 120 mg/dl", "value": "1"},
            ]),
        ], className='mb-3')

restecg_div = html.Div([
                dbc.Label("Resultado da ECG em repouso:"),
                dbc.Select(id="restecg", options=[
                    {"label": "Normal", "value": "0"},
                    {"label": "Anormalidade de ST-T", "value": "1"},
                    {"label": "Hipertrofia Ventricular esquerda", "value": "2"},
                ])
            ])

thalach_div = html.Div([
                dbc.Label("Máximo heart rate alcançado:"),
                dbc.Input(id="thalach", type="number", placeholder="Digite o valor"),
            ])

exang_div = html.Div([
                dbc.Label("Angina induzida pelo exercício:"),
                dbc.Select(id="exang", options=[
                    {"label": "Não", "value": "0"},
                    {"label": "Sim", "value": "1"},
                ])
            ])

oldpeak_div = html.Div([
                dbc.Label("Depressão ST induzida pelo exercício:"),
                dbc.Input(id="oldpeak", type="number", placeholder="Digite o valor"),
            ])

slope_div = html.Div([
                dbc.Label("Inclinação do segmento ST:"),
                dbc.Select(id="slope", options=[
                    {"label": "Ascendente", "value": "1"},
                    {"label": "Plano", "value": "2"},
                    {"label": "Descendente", "value": "3"},
                ])          
            ])


ca_div = html.Div([
                dbc.Label("Número de vasos principais coloridos:"),
                dbc.Select(id="ca", options=[
                    {"label": "0", "value": "0"},
                    {"label": "1", "value": "1"},
                    {"label": "2", "value": "2"},
                    {"label": "3", "value": "3"},
                ])
            ])

thal_div = html.Div([
                dbc.Label("Thal:"),
                dbc.Select(id="thal", options=[
                    {"label": "Normal", "value": "3"},
                    {"label": "Fixed defect", "value": "6"},
                ])
            ])

submit_div = html.Div([
                dbc.Button("Prever", color="success", id="submit-button", n_clicks=0),
            ], className='mb-5')

predict_div = html.Div(id="predict")