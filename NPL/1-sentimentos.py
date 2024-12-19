import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

def get_bag_of_words(data, max_features = 50):
    vetorizar = CountVectorizer(max_features=max_features)

    bag_of_words = vetorizar.fit_transform(data)
    print(bag_of_words.shape)

    matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns = vetorizar.get_feature_names_out())
    print(matriz_esparsa)
    return matriz_esparsa

if __name__ == "__main__":
    #load data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "data/dataset_avaliacoes.csv")
    data = get_csv_data(data_path)
    print(data.head())

    y_label = "sentimento"
    x, y = get_model_data(data, y_label)
    print(x.head())
    print(y.head())

    #transform data
    y = y.replace({'positivo': 1, 'negativo': 0})
    print(y.head())

    matriz = get_bag_of_words(x['avaliacao'])

    random_seed = 4978
    x_train, x_test, y_train, y_test = train_test_split(matriz, y, random_state = random_seed, stratify=y)

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(x_train, y_train)
    accuracy = regressao_logistica.score(x_test, y_test)
    print(f'Accuracy: {accuracy * 100}')