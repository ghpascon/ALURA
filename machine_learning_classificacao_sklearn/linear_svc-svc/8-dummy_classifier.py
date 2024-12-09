import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

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

def compare_to_dummys(train_x, train_y, test_x, test_y, accuracy):
    """
    Compara o modelo fornecido com vários modelos DummyClassifier usando diferentes estratégias.
    
    Args:
        train_x: Dados de treinamento (features).
        train_y: Rótulos de treinamento.
        test_x: Dados de teste (features).
        test_y: Rótulos de teste.
        accuracy: Acurácia do modelo fornecido.
    """
    strategies = ["most_frequent", "stratified", "uniform", "prior"]
    results = {}

    print(f"Acurácia do modelo fornecido: {accuracy * 100:.2f}%\n")
    print("Comparação com modelos Dummy:")
    
    for strategy in strategies:
        # Criando e treinando o modelo dummy
        dummy_model = DummyClassifier(strategy=strategy)
        dummy_model.fit(train_x, train_y)
        
        # Fazendo previsões
        dummy_predictions = dummy_model.predict(test_x)
        
        # Calculando a acurácia
        dummy_accuracy = accuracy_score(test_y, dummy_predictions)
        results[strategy] = dummy_accuracy
        
        print(f" - Estratégia '{strategy}': {dummy_accuracy * 100:.2f}%")
    
    # Comparando com a acurácia do modelo fornecido
    superior_strategies = [strategy for strategy, acc in results.items() if accuracy > acc]
    inferior_strategies = [strategy for strategy, acc in results.items() if accuracy <= acc]

    print("\nResultado da comparação:")
    if superior_strategies:
        print(f" - O modelo fornecido superou as estratégias: {', '.join(superior_strategies)}")
    if inferior_strategies:
        print(f" - O modelo fornecido foi igualado ou superado pelas estratégias: {', '.join(inferior_strategies)}")
    else:
        print(" - O modelo fornecido é superior a todos os modelos Dummy.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/precos.csv")
    data = get_csv_data(data_path)

    #normalizar medidas
    data["km_por_ano"] = data["milhas_por_ano"] * 1.60934
    data = data.drop(columns=["milhas_por_ano"])

    data["idade"] = datetime.today().year - data["ano_do_modelo"]
    data = data.drop(columns=["ano_do_modelo"])

    print(data.head())
    
    y_label = "vendido"
    x, y = get_model_data(data, y_label)
    
    seed = 20
    test_size=0.2
    train_x,test_x,train_y,test_y=train_test_split(x, y,test_size=test_size, random_state=seed, stratify=y)

    # Padronizando os dados
    scaler = StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    model = SVC(gamma='auto')
    model.fit(train_x, train_y)
    
    predictions = model.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    print(f'Acuracy: {accuracy*100}%')

    compare_to_dummys(train_x, train_y, test_x, test_y, accuracy)


