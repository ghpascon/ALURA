'''
PE como y
'''

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score 

import statsmodels.api as sm


from statsmodels.stats.outliers_influence import variance_inflation_factor

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

def train_models(raw_train_x, raw_test_x, train_y, test_y, plot = False):
    """
    Treina diversos modelos de regressão linear com diferentes conjuntos de colunas.
    Retorna o melhor modelo baseado na acurácia (R²).
    """
    # Lista de colunas selecionadas para os modelos
    selected_columns = [
        [
            "AT", 
            "V", 
            "AP",
            "RH"
        ],
        [
            "AT", 
            "V"
        ],
        [
            "AT", 
            "V", 
            "AP"
        ]
    ]

    models = []       # Lista para armazenar os modelos
    accuracies = []   # Lista para armazenar as acurácias
    predictions = []  # Lista para armazenar as previsões

    # Itera sobre cada conjunto de colunas
    for i, cols in enumerate(selected_columns):
        # Seleciona as colunas e adiciona constante
        train_x = sm.add_constant(raw_train_x[cols])
        test_x = sm.add_constant(raw_test_x[cols])

        # Treina o modelo
        model = sm.OLS(train_y, train_x).fit()
        models.append(model)

        # Verifica a multicolinearidade <5 -> sem multicolinearidade
        vif = pd.DataFrame()
        vif['variaveis'] = ["const"] + cols  # Adding 'const' as the first column for intercept
        vif['vif'] = [variance_inflation_factor(train_x.values, i) for i in range(len(cols) + 1)]
        print(f'VIF para modelo {i}:', vif)

        print(model.summary())

        # Faz previsões e calcula a acurácia
        preds = model.predict(test_x)
        predictions.append(preds)

        acc = r2_score(test_y, preds)
        accuracies.append(acc)

        print(f'Acurácia do modelo_{i}: {acc*100:.2f}%')

        if plot:
            plot_predictions_vs_real_prices(test_y, preds)
            plot_residuals(test_y, preds)  

    # Identifica o melhor modelo com base na acurácia
    best_index = np.argmax(accuracies)  # You can adjust this if necessary
    best_model = models[best_index]

    print(f'\nMelhor modelo: modelo_{best_index} com acurácia {accuracies[best_index]*100:.2f}%')
    
    return best_model, accuracies[best_index], predictions[best_index]

def prever_preco(model, house):
    return model.predict(house)

def plot_predictions_vs_real_prices(test_y, new_houses_predictions):
    """
    Cria um gráfico de dispersão para comparar preços reais e previstos.
    """
    plt.figure(figsize=(10, 5))
    plt.scatter(test_y, new_houses_predictions, color='blue', alpha=0.7)
    plt.plot([min(test_y), max(test_y)], [min(test_y), max(test_y)], 'k--', lw=2, color='red') # Linha de referência
    plt.xlabel('PE Real')
    plt.ylabel('PE Previsto')
    plt.title('PE Real vs PE Previsto')
    plt.show()

def plot_residuals(real_prices, predicted_prices):
    """
    Cria um gráfico de dispersão para visualizar os resíduos.
    """
    residuals = real_prices - predicted_prices
    plt.figure(figsize=(10, 5))
    plt.scatter(predicted_prices, residuals, color='green', alpha=0.7)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('PE')
    plt.ylabel('Resíduos')
    plt.title('Resíduos vs PE Previsto')
    plt.show()

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/usina.csv")
    data = get_csv_data(data_path)
    print(data.head())

    y_label = "PE"

    #correlação
    corr = data.corr()
    print(corr[y_label])
    
    # plot_heatmap(corr)

    # x_axis = 'AT'
    # y_axis = y_label

    # plot_correlation(data,x_axis,y_axis)

    x, y = get_model_data(data, y_label)
    print(x.head())
    print(y.head())

    # Split em treino e teste
    seed = 10
    test_size=0.25

    #nao utilizar stratify pois não é classificação
    raw_train_x,raw_test_x,train_y,test_y=train_test_split(x, y,test_size=test_size, random_state=seed)

    model, accuracy, predictions = train_models(raw_train_x,raw_test_x,train_y,test_y, plot = False)

    print(model.params)

    new_x = pd.DataFrame({
        "const": [1],
        "AT": [15],
        "V": [42],
        "AP": [1020],
        "RH": [73]
    })

    print(prever_preco(model, new_x)[0])

    