from dash_components import form_components, fig_components
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc

graficos =  dbc.Container([
        dbc.Row([
            fig_components.div_fig_title
        ]),

        dbc.Row([
            fig_components.div_hist_idade
        ]),

        dbc.Row([
            fig_components.div_box_idade
        ]),
    ])

formulario  = dbc.Container([
    dbc.Row([
        form_components.title_div,
    ]),
    dbc.Row([
        dbc.Col([
            form_components.age_div,
            form_components.sex_div,
            form_components.chest_pain_div,
            form_components.trest_bps_div,
            form_components.chol_div,
            form_components.fbs_div,
        ]),
        dbc.Col([
            form_components.restecg_div,
            form_components.thalach_div,
            form_components.exang_div,
            form_components.oldpeak_div,
            form_components.slope_div,
            form_components.ca_div,
            form_components.thal_div,
        ]),
    ]),

    dbc.Row([
        form_components.submit_div,
    ]),

    dbc.Row([
        form_components.predict_div,
    ])        
])