'''
valor da sillhouete varia entre -1 e 1, quanto mais proximo de 1 melhor
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score 
from sklearn.preprocessing import OneHotEncoder

from sklearn.cluster import KMeans 

import pandas as pd

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

def encoder_data(data):
#String para int
    encoder = OneHotEncoder(categories=[['F', 'M', 'NE']], sparse_output=False) 
    
    sexo_encoded = encoder.fit_transform(data[['sexo']])

    endcoded_data = pd.DataFrame(sexo_encoded, columns=encoder.get_feature_names_out(['sexo']))
    
    data = data.drop(columns=['sexo'])
    data = pd.concat([data, endcoded_data], axis=1)
    print(data.head())    
    # Salvar encoder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nome_arquivo = os.path.join(script_dir, "obj/encoder.pkl")
    with open(nome_arquivo, 'wb') as arquivo:
        pickle.dump(encoder, arquivo)
    return data


def create_model(data, n_clusters = 2, random_state=45):
    #model KMeans
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')

    model.fit(data)

    #metrics
    inertia = model.inertia_

    predict =  model.predict(data)
    silhouette = silhouette_score(data, predict)

    return inertia, silhouette

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/dados_mkt.csv")
    data = get_csv_data(data_path)
    print(data.head())

    print(data.info())
    
    data = encoder_data(data)   

    k_list = []
    inertia = []
    silhouette = []
    for k in range(2, 7):
        print(f'Training model with {k} clusters')
        current_inertia, current_silhouette = create_model(
            data=data,
            n_clusters=k,
            random_state=45, 
            )
        k_list.append(k)
        inertia.append(current_inertia)
        silhouette.append(current_silhouette)

    best_inertia = min(inertia)
    best_inertia_idx = inertia.index(best_inertia)
    
    print(f"Inertia: {inertia}")
    print(f"Menor Inercia: {best_inertia}, with {k_list[best_inertia_idx]} clusters")

    best_silhouette = max(silhouette)
    best_silhouette_idx = silhouette.index(best_silhouette)

    print(f"Silhouette: {silhouette}")
    print(f"Maior Silhouette: {best_silhouette}, with {k_list[best_silhouette_idx]} clusters")
    
    