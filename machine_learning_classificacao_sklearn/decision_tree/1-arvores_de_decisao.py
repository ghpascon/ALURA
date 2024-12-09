import pandas as pd

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
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

def plot_decision_tree(model):
    '''
    Necessario adicionar dot as variaveis de ambiente para funcionar
    '''
    tree_struct = export_graphviz(
        decision_tree = model,
        filled=True,
        rounded=True,
        feature_names=x.columns,
        class_names=["Nao vendido", "Vendido"],
        )
    tree_graph = graphviz.Source(tree_struct)
    tree_graph.render("decision_tree", format="png", cleanup=True)  # Salvar como PNG
    tree_graph.view()

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
    
    seed = 15
    test_size=0.2
    raw_train_x,raw_test_x,train_y,test_y=train_test_split(x, y,test_size=test_size, random_state=seed, stratify=y)

    #parametrizar
    max_depth = 5
    raw_model = DecisionTreeClassifier(max_depth=max_depth)
    raw_model.fit(raw_train_x, train_y)
    
    raw_predictions = raw_model.predict(raw_test_x)
    raw_accuracy = accuracy_score(test_y, raw_predictions)
    print(f'Acuracy from raw_model: {raw_accuracy*100}%')

    # # Padronizando os dados
    scaler = StandardScaler()
    scaler.fit(raw_train_x)
    train_x = scaler.transform(raw_train_x)
    test_x = scaler.transform(raw_test_x)
    
    # Treinando o modelo novamente com os dados padronizados
    scaler_model = DecisionTreeClassifier(max_depth=max_depth)
    scaler_model.fit(raw_train_x, train_y)
    
    scaler_predictions = scaler_model.predict(test_x)
    scaler_accuracy = accuracy_score(test_y, scaler_predictions)
    print(f'Acuracy from scaler_model: {scaler_accuracy*100}%')

    model = raw_model if raw_accuracy >= scaler_accuracy else scaler_model 
    plot_decision_tree(model)

    accuracy = max(raw_accuracy, scaler_accuracy)
    compare_to_dummys(train_x, train_y, test_x, test_y, accuracy)


