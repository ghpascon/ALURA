import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np

def get_model_data(data, y_label):
    if data.empty:
        print("Os dados estão vazios.")
        return None,None
    
    x = data.drop(columns=[y_label])
    y = data[y_label]
    
    return x, y

def plot_all_data(data):
    plt.figure(figsize=(10, 6))  # Tamanho do gráfico

    # Gráfico de linha
    sns.lineplot(x='tempo', y='passageiros', data=data, label='Dado Completo', marker='o')

    # Personalizações
    plt.title('Número de Passageiros ao Longo do Tempo', fontsize=16)
    plt.xlabel('Tempo', fontsize=14)
    plt.ylabel('Número de Passageiros', fontsize=14)
    plt.legend(loc='upper left')  # Posição da legenda
    plt.xticks(rotation=45)  # Rotacionar os meses no eixo X
    plt.tight_layout()

    # Exibir o gráfico
    plt.show()    

def scale_data(data):
    sc = StandardScaler()
    sc.fit(data)  # Ajusta o escalador aos dados
    sc_data = sc.transform(data)  # Transforma os dados

    # Retorna os dados escalados como DataFrame com os mesmos nomes de colunas
    return sc_data, sc


def plot_predict(x_train, y_train, train_predict):
    """
    Plota os valores reais de treinamento interligados com uma linha e a linha de previsões do modelo em cores diferentes.

    Args:
        x_train (pd.DataFrame or np.array): Dados de entrada usados no treinamento.
        y_train (pd.Series or np.array): Valores reais do conjunto de treinamento.
        train_predict (np.array): Valores previstos pelo modelo.
    """
    plt.figure(figsize=(10, 6))  # Configura o tamanho do gráfico

    # Plota os valores reais interligados por uma linha
    sns.lineplot(
        x=x_train.index,
        y=y_train,
        label="Valores Reais",
        color="blue",
        linewidth=2  # Largura da linha
    )
    
    # Adiciona os pontos dos valores reais
    sns.scatterplot(
        x=x_train.index,
        y=y_train,
        color="blue",
        s=50  # Tamanho dos pontos
    )

    # Traça a linha das previsões
    sns.lineplot(
        x=x_train.index,
        y=train_predict.flatten(),
        label="Previsões",
        color="red",
        linewidth=2  # Largura da linha
    )

    # Personalizações
    plt.title("Valores Reais e Previsões do Modelo", fontsize=16)
    plt.xlabel("Índice", fontsize=14)
    plt.ylabel("Valores", fontsize=14)
    plt.legend(loc="upper left")  # Posição da legenda
    plt.tight_layout()  # Ajuste do layout para evitar cortes

    # Exibe o gráfico
    plt.show()


def plot_all_predictions(x_train, x_test, y_train, y_test, train_predict, test_predict, sc, columns):
    # Create DataFrames for the original and predicted values
    df_train = pd.DataFrame(np.column_stack((x_train[:, -1], y_train)), columns=columns)
    df_test = pd.DataFrame(np.column_stack((x_test[:, -1], y_test)), columns=columns)

    # Para as previsões
    df_pred_train = pd.DataFrame(np.column_stack((x_train[:, -1], train_predict)), columns=columns)
    df_pred_test = pd.DataFrame(np.column_stack((x_test[:, -1], test_predict)), columns=columns)
    
    train_indices = np.arange(len(df_train))
    test_indices = np.arange(len(df_train), len(df_train) + len(df_test))

    # Assuming the first column in the DataFrame corresponds to the feature (e.g., time step) and second to target values
    plt.figure(figsize=(12, 6))

    # Plot for the training set (using feature values from the first column as the x-axis)
    sns.lineplot(x=train_indices, y=df_train.iloc[:, 1], label='Original Train', color='blue')
    sns.lineplot(x=train_indices, y=df_pred_train.iloc[:, 1], label='Predicted Train', color='red')
    
    # Plot for the test set (using feature values from the first column as the x-axis)
    sns.lineplot(x=test_indices, y=df_test.iloc[:, 1], label='Original Test', color='green')
    sns.lineplot(x=test_indices, y=df_pred_test.iloc[:, 1], label='Predicted Test', color='orange')

    # Labels and Title
    plt.title('Original vs Predicted Values')
    plt.xlabel('Feature / Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

def passageiro_in_out(data):
    # Shift para ajustar a coluna 'tempo' com base na coluna 'passageiros'
    data['tempo'] = data['passageiros'].shift(1)
    
    # Excluir a primeira linha
    data = data.iloc[1:].reset_index(drop=True)
    
    return data
    
def separa_dados(vetor,n_passos):
    """Entrada: vetor: número de passageiros
                n_passos: número de passos no regressor
        Saída:
                X_novo: Array 2D 
                y_novo: Array 1D - Nosso alvo
    """
    X_novo, y_novo = [], []
    for i in range(n_passos,vetor.shape[0]):
        X_novo.append(list(vetor.loc[i-n_passos:i-1]))
        y_novo.append(vetor.loc[i])
    X_novo, y_novo = np.array(X_novo), np.array(y_novo) 
    return X_novo, y_novo

if __name__ == "__main__":
    data = pd.read_csv('keras_previsao/data/passageiros.csv')
    
    # plot_all_data(data)

    data,sc = scale_data(data)

    # plot_all_data(data)
    x=data[:,0] #Features - Características - Tempo
    y=data[:,1] #Alvo - Número de passageiros

    #separar em treino e teste
    tamanho_treino = int(len(y)*0.8)
    tamanho_teste = len(y)-tamanho_treino 

    x_train = x[0:tamanho_treino]
    y_train = y[0:tamanho_treino]
    x_test = x[tamanho_treino:len(x)]
    y_test = y[tamanho_treino:len(y)]

    print(y_train)
    print(y_test)

    passos = 1
    x_train, y_train = separa_dados(pd.DataFrame(y_train)[0],passos)
    x_test, y_test = separa_dados(pd.DataFrame(y_test)[0],passos)
    print('\n\ndados separados\n\n')
    print(x_train)
    print(x_test)

    #preparar o modelo com regressão
    model = keras.models.Sequential()

    #camada de entrada
    model.add(keras.layers.Dense(
        units=8,  
        input_dim=passos, 
        kernel_initializer='ones',
        use_bias=False,
        activation='linear',
        )
    )

    #camada oculta_1
    model.add(keras.layers.Dense(
        units=64,
        kernel_initializer='random_uniform',
        use_bias=False,
        activation='sigmoid',
        )
    )

    #camada de saida
    model.add(keras.layers.Dense(
        units=1,
        kernel_initializer='random_uniform',
        use_bias=False,
        activation='linear',
        )
    )
    
    #mean_squared_error -> regressão linear
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        # metrics=['mean_absolute_error']
        )

    model.summary()

    epochs = 100
    model.fit(x_train, y_train, epochs=epochs)

    train_predict = model.predict(x_train)

    # plot_predict(x_train, y_train, train_predict)

    #previsão para o teste
    test_predict = model.predict(x_test)

    # plot_predict(x_test, y_test, test_predict)

    plot_all_predictions(x_train, x_test, y_train, y_test, train_predict, test_predict, sc, ['0', '1'])
    
    