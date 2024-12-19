import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import tokenize, FreqDist, corpus
import seaborn as sns

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

def get_bag_of_words(data, max_features=50):
    """
    Gera a matriz bag-of-words a partir de uma lista de frases.

    Parâmetros:
    - data: lista de strings contendo as avaliações.
    - max_features: número máximo de palavras únicas a serem consideradas.

    Retorno:
    - Matriz esparsa como DataFrame.
    """

    # Garante que `data` seja uma lista de strings
    if isinstance(data, pd.Series):
        data = data.tolist()

    # Inicializa o vetorizador com o limite de max_features
    vetorizar = CountVectorizer(max_features=max_features)

    # Gera a matriz bag-of-words
    bag_of_words = vetorizar.fit_transform(data)
    print(bag_of_words.shape)

    # Converte a matriz esparsa para um DataFrame
    matriz_esparsa = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns=vetorizar.get_feature_names_out())
    print(matriz_esparsa)

    return matriz_esparsa


def generate_wordcloud(data, y):
    good_avaliations = data[y != 0]
    good_words = ' '.join([txt for txt in good_avaliations])

    good_cloud = WordCloud(width=800, height=500, max_font_size=100, collocations=False).generate(good_words)
    plt.figure(figsize=(10, 10))
    plt.title('Good Avaliations')
    plt.imshow(good_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    bad_avaliations = data[y == 0]
    bad_words =''.join([txt for txt in bad_avaliations])
    bad_cloud = WordCloud(width=800, height=500, max_font_size=100, collocations=False).generate(bad_words)
    plt.figure(figsize=(10, 10))
    plt.title('Bad Avaliations')
    plt.imshow(bad_cloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def data_treatment(data, token):
    irrelevant_words = corpus.stopwords.words('portuguese')
    frases_processadas = []

    for opiniao in data:
        palavras_textos = token.tokenize(opiniao.lower())
        nova_frase = [palavra for palavra in palavras_textos if palavra not in irrelevant_words]
        frases_processadas.append(' '.join(nova_frase))      

    return frases_processadas  

def word_tokenization(data):
    token = tokenize.WhitespaceTokenizer()
    
    data = data_treatment(data, token)
    all_words =''.join([txt for txt in data])
     
    token_words = token.tokenize(all_words)

    words_freq = FreqDist(token_words)
    df_freq = pd.DataFrame({'Palavra': list(words_freq.keys()), 'Frequencia': list(words_freq.values())})
    df_freq = df_freq.sort_values(by='Frequencia', ascending=False).reset_index(drop=True)
    print(df_freq)
    return df_freq, data

def plot_freq(df_freq, samples=20):
    """
    Plota as frequências das palavras como um gráfico de barras.
    
    Parâmetros:
    - df_freq: DataFrame contendo as palavras e suas frequências.
    - samples: Número de palavras a serem exibidas (por padrão, 20).
    """
    # Garante que o DataFrame esteja ordenado
    df_freq = df_freq.sort_values(by='Frequencia', ascending=False)
    
    # Configura o tamanho da figura
    plt.figure(figsize=(10, 6))
    
    # Cria o gráfico de barras
    ax = sns.barplot(data=df_freq[:samples], x='Palavra', y='Frequencia', color='#ff0000')
    
    # Configura os rótulos e o título
    ax.set_ylabel('Contagem')
    ax.set_xlabel('Palavra')
    ax.set_title(f'Top {samples} Palavras Mais Frequentes')
    
    # Exibe o gráfico
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

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

    avaliacao = x['avaliacao']
    # generate_wordcloud(avaliacao, y)
    df_freq, all_words = word_tokenization(avaliacao)
    plot_freq(df_freq, 10)

    # matriz = get_bag_of_words(avaliacao)
    matriz = get_bag_of_words(all_words)

    #treinamento do modelo de regressão logística
    random_seed = 4978
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(matriz, y, random_state = random_seed, stratify=y, test_size=test_size)

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(x_train, y_train)
    accuracy = regressao_logistica.score(x_test, y_test)
    print(f'Accuracy: {accuracy * 100}')