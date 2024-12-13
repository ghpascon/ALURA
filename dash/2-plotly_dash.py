from ucimlrepo import fetch_ucirepo
import plotly.express as px
from dash import Dash, dcc, html
import pickle
import os

def get_dataframe():
    heart_disease = fetch_ucirepo(id=45)
    
    # Data (as pandas DataFrames)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Variable information
    print(heart_disease.variables)

    return X, y

def dash_page(fig_list):
    app = Dash(__name__)

    app.layout = html.Div([
        
        html.Div([
            html.H1("Histograma da Idade"),
            dcc.Graph(figure=fig_list[0]),
        ]),
        
        html.Div([
            html.H1("BOXPLOT da Idade"),
            dcc.Graph(figure=fig_list[1]),
        ])      

    ])

    app.run_server(debug=True)

def save_fig_list(fig_list):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nome_arquivo = os.path.join(script_dir, "figs.pkl")
    with open(nome_arquivo, 'wb') as arquivo:
        pickle.dump(fig_list, arquivo)

if __name__ == "__main__":
    X, y = get_dataframe()
    data = X

    # Separate between 'yes' or 'no' for the chance of disease
    y = (y > 0).astype(int)

    data["doenca"] = y

    print(X.head())
    print(y.head())

    # figs
    fig_0 = px.histogram(
        data,
        x="age",
        nbins=30,
        title="Histograma da Idade",
        color="doenca",
    )
    fig_0.update_traces(marker=dict(line=dict(color="black", width=1)))

    fig_1 = px.box(data, x="doenca", y = "age", title="BOX PLOT", color="doenca")

    # Store the figure in a list
    fig_list = [fig_0, fig_1]

    save_fig_list(fig_list)

    # Display the histogram in a Dash app
    dash_page(fig_list)
