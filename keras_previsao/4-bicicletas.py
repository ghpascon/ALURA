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
    sc.fit(data['contagem'].values.reshape(-1,1))  # Ajusta o escalador aos dados
    sc_data = sc.transform(data['contagem'].values.reshape(-1,1))  # Transforma os dados

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
    return X_novo.reshape((X_novo.shape[0],X_novo.shape[1],1)), y_novo

def plot_results(results):
    #plotar resultados
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    data_original = pd.read_csv('keras_previsao/data/bicicletas.csv')
    
    data,sc = scale_data(data_original)

    #separar em treino e teste
    tamanho_treino = int(len(data)*0.8)
    tamanho_teste = len(data)-tamanho_treino 

    y_train = data[0:tamanho_treino]
    y_test = data[tamanho_treino:len(data)]

    passos = 5
    x_train, y_train = separa_dados(pd.DataFrame(y_train)[0],passos)
    x_test, y_test = separa_dados(pd.DataFrame(y_test)[0],passos)
    print('\n\ndados separados\n\n')
    print(x_train)
    print(x_test)


    #preparar o modelo com regressão
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(
        units=128,
        input_shape=(x_train.shape[1], x_train.shape[2]), 
        )
    )

    model.add(keras.layers.Dense(
            units=1
        )
    )
    
    #mean_squared_error -> regressão linear
    model.compile(
        optimizer='RMSProp',
        loss='mean_squared_error',
        # metrics=['mean_absolute_error']
        )

    model.summary()

    epochs = 10
    results = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs)
    
    plot_results(results)

    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)

    # Alinhar dimensões: Reduzir train_predictions e test_predictions para 1D, se necessário
    if train_predictions.ndim > 1:
        train_predictions = train_predictions.ravel()

    if test_predictions.ndim > 1:
        test_predictions = test_predictions.ravel()

    # Criar DataFrames para treino com índice
    train_data = pd.DataFrame({
        "Índice": np.arange(len(y_train)),
        "Tipo": "Treino",
        "Valor": y_train
    })

    train_predictions_data = pd.DataFrame({
        "Índice": np.arange(len(train_predictions)),
        "Tipo": "Previsão Treino",
        "Valor": train_predictions
    })

    # Criar DataFrames para teste com índice deslocado para a direita
    test_data = pd.DataFrame({
        "Índice": np.arange(len(y_test)) + len(y_train),  # Desloca o índice do teste após o treino
        "Tipo": "Teste",
        "Valor": y_test
    })

    test_predictions_data = pd.DataFrame({
        "Índice": np.arange(len(test_predictions)) + len(train_predictions),  # Desloca o índice das previsões de teste
        "Tipo": "Previsão Teste",
        "Valor": test_predictions
    })

    # Combinar os dados de treino e teste para um gráfico conjunto
    combined_data = pd.concat([train_data, train_predictions_data, test_data, test_predictions_data])

    # Criar o lineplot
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=combined_data, x="Índice", y="Valor", hue="Tipo", palette="Set2", marker="o")

    # Configurações do gráfico
    plt.title("Comparação de Valores Reais e Previstos (Lineplot)")
    plt.ylabel("Valores")
    plt.xlabel("Índice")
    plt.legend(title="Categoria")

    plt.show()