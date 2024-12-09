'''
importante usar o train_test_split para garantir uma proporcionalidade e embaralhar os dados
'''

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import os

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

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/tracking.csv")
    data = get_csv_data(data_path)

    y_label = "comprou"
    x, y = get_model_data(data, y_label)
    
    #definir seed para ficar replicavel
    seed = 3  
    test_size=0.2
    train_x,test_x,train_y,test_y=train_test_split(x, y,test_size=test_size, random_state=seed, stratify=y)
    
    print(train_x.head())
    print(test_x.head())
    print(train_y.head())
    print(test_y.head())

    print(train_y.value_counts())
    print(test_y.value_counts())

    model = LinearSVC()
    model.fit(train_x, train_y)
    
    predictions = model.predict(test_x)
    acuracy = accuracy_score(test_y, predictions)
    print(f'Acuracy: {acuracy*100}%')