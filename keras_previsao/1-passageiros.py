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
    return pd.DataFrame(sc_data, columns=data.columns), sc


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
    df_train = pd.DataFrame(sc.inverse_transform(np.column_stack((x_train, y_train))), columns=columns)
    df_test = pd.DataFrame(sc.inverse_transform(np.column_stack((x_test, y_test))), columns=columns)
    df_pred_train = pd.DataFrame(sc.inverse_transform(np.column_stack((x_train, train_predict))), columns=columns)
    df_pred_test = pd.DataFrame(sc.inverse_transform(np.column_stack((x_test, test_predict))), columns=columns)
    
    # Assuming the first column in the DataFrame corresponds to the feature (e.g., time step) and second to target values
    plt.figure(figsize=(12, 6))

    # Plot for the training set (using feature values from the first column as the x-axis)
    sns.lineplot(x=df_train.iloc[:, 0], y=df_train.iloc[:, 1], label='Original Train', color='blue')
    sns.lineplot(x=df_pred_train.iloc[:, 0], y=df_pred_train.iloc[:, 1], label='Predicted Train', color='red')
    
    # Plot for the test set (using feature values from the first column as the x-axis)
    sns.lineplot(x=df_test.iloc[:, 0], y=df_test.iloc[:, 1], label='Original Test', color='green')
    sns.lineplot(x=df_pred_test.iloc[:, 0], y=df_pred_test.iloc[:, 1], label='Predicted Test', color='orange')

    # Labels and Title
    plt.title('Original vs Predicted Values')
    plt.xlabel('Feature / Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    data = pd.read_csv('keras_previsao/data/passageiros.csv')
    
    # plot_all_data(data)

    data,sc = scale_data(data)

    # plot_all_data(data)

    y_label = "passageiros"
    x, y = get_model_data(data, y_label)
    print (x.head())
    print (y.head())

    #separar em treino e teste
    random_seed = 11
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = random_seed, test_size=0.2, shuffle=False)

    #preparar o modelo com regressão
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(
        units=1,  # só uma saída (número de passageiros)
        input_dim=x_train.shape[1], 
        kernel_initializer='Ones',
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

    model.fit(x_train, y_train)

    train_predict = model.predict(x_train)

    # plot_predict(x_train, y_train, train_predict)

    #previsão para o teste
    test_predict = model.predict(x_test)

    # plot_predict(x_test, y_test, test_predict)

    plot_all_predictions(x_train, x_test, y_train, y_test, train_predict, test_predict, sc, data.columns)
    
    