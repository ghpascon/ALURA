import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

def plot_price_time(data):
    sns.scatterplot(x="horas_esperadas", y="preco",hue="finalizado", data=data)
    plt.title("Relação entre Horas Esperadas e Preço")
    plt.xlabel("Horas Esperadas")
    plt.ylabel("Preço")
    plt.show()

    sns.relplot(x="horas_esperadas", y="preco",hue="finalizado", col="finalizado", data=data)
    plt.title("Relação entre Horas Esperadas e Preço")
    plt.xlabel("Horas Esperadas")
    plt.ylabel("Preço")
    plt.show()

def plot_results(test_x, test_y):
#plotar resultados
    x_min=test_x[:, 0].min()
    x_max=test_x[:, 0].max()
    y_min=test_x[:, 1].min()
    y_max=test_x[:, 1].max()
    print(x_min)
    print(x_max)
    print(y_min)
    print(y_max)

    pixels = 100                            

    eixo_x = np.arange(x_min, x_max, (x_max-x_min) / pixels)
    eixo_y = np.arange(y_min, y_max, (y_max-y_min) / pixels)
    
    xx, yy = np.meshgrid(eixo_x, eixo_y)

    dots = np.c_[xx.ravel(), yy.ravel()]

    z = model.predict(dots)
    z = z.reshape(xx.shape)

    plt.contour(xx, yy, z)
    plt.scatter(test_x[:, 0], test_x[:, 1], c = test_y, s=7)
    plt.title("Relação entre Horas Esperadas e Preço")
    plt.xlabel("Horas Esperadas")
    plt.ylabel("Preço")
    plt.show()
    

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/projects.csv")
    data = get_csv_data(data_path)
    data["finalizado"] = data["nao_finalizado"].map({1:0, 0:1})
    data = data.drop(columns=["nao_finalizado"])

    print(data.head())
    # plot_price_time(data)

    # Preparando dados
    #retira as horas zeradas
    data = data.query("horas_esperadas > 0")

    y_label = "finalizado"
    x, y = get_model_data(data, y_label)
    
    seed = 21
    test_size=0.2
    train_x,test_x,train_y,test_y=train_test_split(x, y,test_size=test_size, random_state=seed, stratify=y)

    # Padronizando os dados
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    # parametros
    gamma = 'auto'

    model = SVC(gamma=gamma)
    model.fit(train_x, train_y)
    
    predictions = model.predict(test_x)
    acuracy = accuracy_score(test_y, predictions)
    print(f'Acuracy: {acuracy*100}%')

    plot_results(test_x, test_y)