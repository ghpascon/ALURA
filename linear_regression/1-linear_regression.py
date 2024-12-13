'''
Tamanho do primeiro andar para definir o preço da casa
'''

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 

from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler

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

def plot_heatmap(corr):
    mascara = np.zeros_like(corr, dtype=bool)
    mascara[np.triu_indices_from(mascara)] = True

    # Configurar a figura do matplotlib
    f, ax = plt.subplots(figsize=(11, 9))

    # Gerar o mapa de calor (heatmap)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    sns.heatmap(corr, mask=mascara, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .5})

    # Exibir o mapa de calor (heatmap)
    plt.show()

def plot_correlation(data,x_axis,y_axis):
    x = data[x_axis]
    y = data[y_axis]
    
    # Ajuste de uma linha de tendência linear
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    
    plt.scatter(x, y, label='Dados')
    plt.plot(x, poly1d_fn(x), color='red', label='Linha de tendência')
    plt.title(f'Relação entre {x_axis} e {y_axis}')
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()
    plt.show()

def plot_resid(model):
    model.resid.hist()
    plt.title("Histograma de Resid")
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/house_price.csv")
    data = get_csv_data(data_path)
    data = data.drop(columns=["Id"])
    print(data.head())

    #correlação
    corr_label = "preco_de_venda"
    corr = data.corr()
    print(corr[corr_label])
    
    # plot_heatmap(corr)

    x_axis = 'area_primeiro_andar'
    y_axis = 'preco_de_venda'

    # plot_correlation(data,x_axis,y_axis)

    y_label = "preco_de_venda"
    x, y = get_model_data(data, y_label)
    print(x.head())
    print(y.head())

    # Split em treino e teste
    seed = 12
    test_size=0.2

    #nao utilizar stratify pois não é classificação
    raw_train_x,raw_test_x,train_y,test_y=train_test_split(x, y,test_size=test_size, random_state=seed)

    #criar dataframe de treino
    raw_train_df = pd.DataFrame(data = raw_train_x)
    raw_train_df[y_label] = train_y 

    print(raw_train_df.head())

    print()
    print('--------------------------------')
    print()

    raw_model = ols(f'{y_axis} ~ {x_axis}', data = raw_train_df).fit()
    # print(raw_model.params)#intercept 'a', x_axis = 'b'x
    # print(raw_model.summary)
    print(raw_model.rsquared)
    # print(raw_model.resid)#erro em relação a linha de regressão

    # plot_resid(raw_model)

    raw_predictions = raw_model.predict(raw_test_x)
    raw_accuracy = r2_score(test_y, raw_predictions)
    print(f'Acuracy from raw_model: {raw_accuracy*100}%')
