import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

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

def get_model_data(data, train_y_label):
    """
    Divide os dados em treino e teste.
    
    Args:
        data (DataFrame): DataFrame contendo os dados.
        train_y_label (str): Nome da coluna que contém os rótulos de saída.
    
    Returns:
        tuple: (train_x, train_y, test_x, test_y)
    """
    # Verifica se o DataFrame não está vazio
    if data.empty:
        print("Os dados estão vazios.")
        return pd.DataFrame(), pd.Series(), pd.DataFrame(), pd.Series()
    
    # Calcula o índice de separação
    split_index = int(len(data) * 0.75)
    
    # Divide o DataFrame em treino e teste
    train_data = data[:split_index]
    test_data = data[split_index:]
    
    # Separa os dados de entrada e rótulos de saída
    train_x = train_data.drop(columns=[train_y_label])
    train_y = train_data[train_y_label]
    test_x = test_data.drop(columns=[train_y_label])
    test_y = test_data[train_y_label]
    
    return train_x, train_y, test_x, test_y

if __name__ == "__main__":
    data_path = "data/tracking.csv"  # Caminho do arquivo CSV
    data = get_csv_data(data_path)

    if not data.empty:
        print("Dados carregados:")
        print(data.head())

        # Nome da coluna de saída
        train_y_label = "comprou"
        
        # Obtém os dados de treino e teste
        train_x, train_y, test_x, test_y = get_model_data(data, train_y_label)
        
        print("\nDados de entrada para treino (train_x):")
        print(train_x.head())
        
        print("\nRótulos de saída para treino (train_y):")
        print(train_y.head())
        
        print("\nDados de entrada para teste (test_x):")
        print(test_x.head())
        
        print("\nRótulos de saída para teste (test_y):")
        print(test_y.head())
        
        # Treina o modelo
        model = LinearSVC()
        model.fit(train_x, train_y)
        
        predictions = model.predict(test_x)
        acuracy = accuracy_score(test_y, predictions)
        print(f'Acuracy: {acuracy*100}%')

