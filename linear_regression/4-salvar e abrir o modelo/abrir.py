import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 

import statsmodels.api as sm
import pickle


def get_csv_data(path):
    """
    Lê dados de um arquivo CSV usando Pandas e retorna um DataFrame.
    """
    try:
        # Lê o arquivo CSV usando pandas
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {path}")
        return pd.DataFrame()  # Retorna um DataFrame vazio
    except Exception as e:
        print(f"Erro ao ler o arquivo CSV: {e}")
        return pd.DataFrame()  # Retorna um DataFrame vazio
    
def get_model_data(data, y_label):
    if data.empty:
        print("Os dados estão vazios.")
        return None,None
    
    x = data.drop(columns=[y_label])
    y = data[y_label]
    
    return x, y

def prever_preco(model, house):
    return model.predict(house)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "modelos")
    nome_arquivo = os.path.join(model_dir, "modelo_regressao_linear.pkl")
    try:
        with open(nome_arquivo, 'rb') as arquivo:
            model = pickle.load(arquivo)
            print("Modelo carregado com sucesso!")
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        

    new_house = pd.DataFrame({
        "const": [1],
        "area_primeiro_andar": [98],
        "existe_segundo_andar": [0],
        "quantidade_banheiros": [1],
        "qualidade_da_cozinha_Excelente": [1] 
    })

    print(prever_preco(model, new_house)[0])

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "../data/Novas_casas.csv")
    new_houses = get_csv_data(data_path)

    new_houses = new_houses.drop(columns=["Casa"])
    new_houses = sm.add_constant(new_houses)

    print(new_houses.head())

    new_houses_predictions = prever_preco(model, new_houses)
    for idx, price in enumerate(new_houses_predictions):
        print(f"Casa {idx + 1}: Preço previsto = {price}")