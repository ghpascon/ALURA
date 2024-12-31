import pandas as pd
import os

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk import tokenize, FreqDist, corpus, ngrams, RSLPStemmer, download
import seaborn as sns
from unidecode import unidecode

import pickle



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

def get_tfidf(data, max_features=50, ngram_range = (1, 2)):
    '''
    atribuir peso para as palvras
    '''
    # Garante que `data` seja uma lista de strings
    if isinstance(data, pd.Series):
        data = data.tolist()

    # Inicializa o vetorizador com o limite de max_features
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)

    # Gera a matriz bag-of-words
    matriz = tfidf.fit_transform(data)

    # Converte a matriz esparsa para um DataFrame
    matriz_esparsa = pd.DataFrame(matriz.todense(), columns=tfidf.get_feature_names_out())
    print(matriz_esparsa)

    return matriz_esparsa, tfidf

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
    '''
    LowerCase (lower())
    Remover Stopwords (corpus)
    Remover caracteres especiais (unidecode)
    Remover Pontuação e acentos (token e alpha)
    Simplificar a palavra para seu radical (stemmer)
    Utilizar paravras em contexto
    '''
    print('stopwords')
    irrelevant_words = corpus.stopwords.words('portuguese')
    stemmer = RSLPStemmer()

    frases_processadas = []

    for i, opiniao in enumerate(data):
        palavras_textos = token.tokenize(opiniao.lower())

        nova_frase = [stemmer.stem(unidecode(palavra)) for palavra in palavras_textos if palavra not in irrelevant_words and palavra.isalpha()]
        frases_processadas.append(' '.join(nova_frase))      

        if (i + 1) % 100 == 0:  # Mostra progresso a cada 100 iterações
            print(f"Progresso: {i + 1}/{len(data)} opiniões processadas")

    return frases_processadas

def word_tokenization(data):
    token = tokenize.WordPunctTokenizer()
    
    data = data_treatment(data, token)
    all_words =''.join([txt for txt in data])
     
    print('tokenizando as palavras')
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
    print('plotando words')
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
    data = pd.read_csv('NPL/data/imdb.csv')
    print(data.head())

    pt_data = data.drop(columns=['id', 'text_en'])

    print(pt_data.head())

    y_label = "sentiment"
    x, y = get_model_data(pt_data, y_label)
    print(x.head())
    print(y.head())

    #transform data
    y = y.replace({'pos': 1, 'neg': 0})
    print(y.head())
    print(y.value_counts())

    avaliacao = x['text_pt']
    # generate_wordcloud(avaliacao, y)

    df_freq, all_words = word_tokenization(avaliacao)
    plot_freq(df_freq, 10)
    exit()

    # matriz = get_bag_of_words(avaliacao)

    max_features = 1000

    # matriz = get_bag_of_words(all_words, max_features)

    ngram_range = (1, 2)
    matriz, vetor_tfidf = get_tfidf(all_words, max_features, ngram_range)

    #treinamento do modelo de regressão logística
    random_seed = 11
    test_size = 0.2
    x_train, x_test, y_train, y_test = train_test_split(matriz, y, random_state = random_seed, stratify=y, test_size=test_size)

    regressao_logistica = LogisticRegression()
    regressao_logistica.fit(x_train, y_train)

    #pesos
    pesos = pd.DataFrame(
        regressao_logistica.coef_[0].T, 
        index = vetor_tfidf.get_feature_names_out()
    )

    print(pesos.nlargest(50, 0))
    print(pesos.nsmallest(50, 0))


    accuracy = regressao_logistica.score(x_test, y_test)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    #salvar o modelo e o vetor
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, "model")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Cria o diretório "modelos"

    # Salvar o modelo no arquivo
    nome_arquivo = os.path.join(model_dir, "modelo_npl_imdb.pkl")
    with open(nome_arquivo, 'wb') as arquivo:
        pickle.dump(regressao_logistica, arquivo)

    print(f"Modelo salvo em: {nome_arquivo}")

    # Salvar o vetor no arquivo
    nome_arquivo = os.path.join(model_dir, "vetor_npl_imdb.pkl")
    with open(nome_arquivo, 'wb') as arquivo:
        pickle.dump(vetor_tfidf, arquivo)

    print(f"Vetor salvo em: {nome_arquivo}")

    # Salvar a acuracia no arquivo
    nome_arquivo = os.path.join(model_dir, "accuracy_imdb.pkl")
    with open(nome_arquivo, 'wb') as arquivo:
        pickle.dump(accuracy, arquivo)

    print(f"Acuracia salvo em: {nome_arquivo}")

    # Salvar o dataframe de frequencia no arquivo
    nome_arquivo = os.path.join(model_dir, "df_freq_imdb.pkl")
    with open(nome_arquivo, 'wb') as arquivo:
        pickle.dump(df_freq, arquivo)

    print(f"Df_freq salvo em: {nome_arquivo}")

    # Salvar all_words
    nome_arquivo = os.path.join(model_dir, "all_words_imdb.pkl")
    with open(nome_arquivo, 'wb') as arquivo:
        pickle.dump(all_words, arquivo)

    print(f"all_words salvo em: {nome_arquivo}")