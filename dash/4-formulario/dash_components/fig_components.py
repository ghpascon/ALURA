from ucimlrepo import fetch_ucirepo
import plotly.express as px
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

import pickle
import os
    
def get_fig_list():    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nome_arquivo = os.path.join(script_dir, "figs.pkl")
    with open(nome_arquivo, 'rb') as arquivo:
        figs = pickle.load(arquivo)
        return figs
        
fig_list = get_fig_list()

div_fig_title= html.Div([
            html.H1("Graficos relacionados a Doenças Cardiacas pela idade."),
        ],className='text-center mt-5')

div_hist_idade= html.Div([
            html.H2("Histograma da Idade"),
            dcc.Graph(figure=fig_list[0]),
        ])
        
div_box_idade = html.Div([
            html.H2("BOXPLOT da Idade"),
            dcc.Graph(figure=fig_list[1]),
        ]) 