'''
valor da sillhouete varia entre -1 e 1, quanto mais proximo de 1 melhor
'''

from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm 

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

def plot_cotovelo(inercia, k_list):
    plt.figure(figsize=(8, 4))
    plt.plot(k_list, inercia, 'bo-')  # Use diretamente k_list para o eixo x
    plt.xlabel('Número de clusters')
    plt.ylabel('Inércia')
    plt.title('Método do Cotovelo para Determinação de k')
    plt.show()

def graf_silhueta (n_clusters, dados, random_state= 45):

    # Aplica o KMeans ao conjunto de dados
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init = 'auto')
    cluster_previsoes = kmeans.fit_predict(dados)
    
    # Calcula o silhouette score médio
    silhueta_media = silhouette_score(dados, cluster_previsoes)
    print(f'Valor médio para {n_clusters} clusters: {silhueta_media:.3f}')
    
    # Calcula a pontuação de silhueta para cada amostra
    silhueta_amostra = silhouette_samples(dados, cluster_previsoes)
    
    # Configuração da figura para o gráfico de silhueta
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(9, 7)
    
    # Limites do gráfico de silhueta
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(dados) + (n_clusters + 1) * 10])
    
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhueta_amostra = silhueta_amostra[cluster_previsoes == i]
        ith_cluster_silhueta_amostra.sort()
        
        tamanho_cluster_i = ith_cluster_silhueta_amostra.shape[0]
        y_upper = y_lower + tamanho_cluster_i
        
        cor = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhueta_amostra, 
            facecolor=cor, edgecolor=cor, alpha=0.7)

        ax1.text(-0.05, y_lower + 0.5 * tamanho_cluster_i, str(i))
        y_lower = y_upper + 10  # 10 para o espaço entre gráficos
        
    # Linha vertical para a média do Silhouette Score
    ax1.axvline(x=silhueta_media, color='red', linestyle='--')
        
    ax1.set_title(f'Gráfico da Silhueta para {n_clusters} clusters')
    ax1.set_xlabel('Valores do coeficiente de silhueta')
    ax1.set_ylabel('Rótulo do cluster')
    
    ax1.set_yticks([])  # Remove os ticks do eixo y
    ax1.set_xticks([i/10.0 for i in range(-1, 11)])
    
    plt.show()

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
    for k in range(2, 21):
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
    
    plot_cotovelo(inercia=inertia, k_list=k_list)

    graf_silhueta(k_list[best_silhouette_idx], data, random_state = 45)