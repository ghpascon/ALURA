import streamlit as st

import pandas as pd
from nltk import tokenize, FreqDist, corpus, RSLPStemmer, download
from unidecode import unidecode
import pickle
import warnings
import plotly.express as px
from wordcloud import WordCloud

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Download NLTK resources if not available
download('stopwords')
download('rslp')

def load_model():
    try:
        with open("NPL/model/modelo_npl.pkl", 'rb') as arquivo:
            model = pickle.load(arquivo)
            print("Modelo carregado com sucesso!")

        with open("NPL/model/vetor_npl.pkl", 'rb') as arquivo:
            vetor = pickle.load(arquivo)
            print("Vetor carregado com sucesso!")

        return model, vetor
    except FileNotFoundError as e:
        print(f"Erro ao carregar arquivos: {e}")
        exit()

class WordToken:
    def __init__(self):
        self.token = tokenize.WordPunctTokenizer()
        self.irrelevant_words = corpus.stopwords.words('portuguese')
        self.stemmer = RSLPStemmer()

    def data_treatment(self, data):
        """
        Process the input data to remove stopwords, punctuation, accents, and reduce words to their stem.
        """
        frases_processadas = []

        for opiniao in data:
            palavras_textos = self.token.tokenize(opiniao.lower())
            nova_frase = [
                self.stemmer.stem(unidecode(palavra))
                for palavra in palavras_textos
                if palavra not in self.irrelevant_words and palavra.isalpha()
            ]
            frases_processadas.append(' '.join(nova_frase))

        return frases_processadas

    def word_tokenization(self, data):
        """
        Tokenize and calculate word frequencies from the input data.
        """
        if isinstance(data, str):  # Ensure data is a list
            data = [data]

        data = self.data_treatment(data)
        all_words = ' '.join(data)

        token_words = self.token.tokenize(all_words)
        words_freq = FreqDist(token_words)
        df_freq = pd.DataFrame({'Palavra': list(words_freq.keys()), 'Frequencia': list(words_freq.values())})
        df_freq = df_freq.sort_values(by='Frequencia', ascending=False).reset_index(drop=True)

        return df_freq, data

def input_predict(texto, modelo, vetor, tokenizer):
    if texto is None: 
        st.write("Coloque uma opinião válida")

    _, processed_data = tokenizer.word_tokenization(texto)

    # Transform the processed data with the vectorizer
    input_vectorized = vetor.transform(processed_data)

    # Make the prediction
    prediction = modelo.predict(input_vectorized)
    probabilities = modelo.predict_proba(input_vectorized)

    # Determine class labels and their probabilities
    class_labels = ["Negativa", "Positiva"]
    prob_data = {
        'Classe': class_labels,
        'Probabilidade': [probabilities[0][0], probabilities[0][1]]
    }

    # Display the results
    st.write(f"\nOpinião: '{texto}' => Previsão: {'Negativa' if prediction[0] == 0 else 'Positiva'} com {max(probabilities[0]) * 100:.2f}% de probabilidade")

    fig = px.bar(prob_data, x='Classe', y='Probabilidade',
                 title=f"Gráfico de Probabilidade",
                 labels={'Probabilidade': 'Probabilidade (%)'},
                 text='Probabilidade')
    st.write(fig)

    # Show the normalized words and their weights
    feature_names = vetor.get_feature_names_out()
    input_words = processed_data[0].split()  # Words in the processed opinion

    word_weights = []
    for word in input_words:
        if word in feature_names:
            index = list(feature_names).index(word)
            weight = modelo.coef_[0][index]
            word_weights.append({'Palavra': word, 'Peso': weight})
        else:
            word_weights.append({'Palavra': word, 'Peso': None})  # If word not found in vocabulary

    df_word_weights = pd.DataFrame(word_weights)

    # Show the DataFrame with normalized words and their weights
    st.write("Peso das palavras normalizadas de acordo com o modelo:")    
    st.write(df_word_weights)    
    

def get_file(name):
    try:
        with open(f"NPL/model/{name}.pkl", 'rb') as arquivo:
            return pickle.load(arquivo)

    except FileNotFoundError as e:
            return  f'Erro ao carregar {name}'


def streamlit_page(modelo, vetor, tokenizer):
    accuracy = get_file('accuracy')
    df_freq = get_file('df_freq')
    #pesos
    pesos = pd.DataFrame(
        modelo.coef_[0].T, 
        index = vetor.get_feature_names_out()
    )

    st.set_page_config(
        page_title="Classificador NPL",
        layout="wide",
        page_icon="NPL/data/alura.png",
        initial_sidebar_state="expanded"
        )

    st.sidebar.title("Menu")
    opcao = st.sidebar.selectbox(
        "Selecione uma funcionalidade",
        ['Prever Opinião', 'Dados do modelo']
    )

    if opcao == "Prever Opinião":
        st.title("Classificador de opinião com base em um modelo NPL")
        st.write("Coloque sua avaliação sobre um produto para o modelo classificar em positiva ou negativa, mostrar um gráfico de probabilidade e o peso das palavras normalizados utilizadas na avaliação")
        
        texto = st.text_input("Avaliação:")
        if texto:
            input_predict(texto, modelo, vetor, tokenizer)

    if opcao == "Dados do modelo":
        st.title("Dados de treino e teste do modelo NPL criado com Sklearn")
        st.divider()

        # Navegador de abas
        tab1, tab2, tab3 = st.tabs(["Acurácia", "Distribuição de Frequências", "Pesos"])

        with tab1:
            st.write(f"Acurácia do modelo NPL no teste foi de: {accuracy*100:.2f}%")
        
        with tab2:
            num = st.slider("Escolha a quantidade de palavras que serão exibidas", 0, 100, 20, key="s0")
            fig_freq = px.bar(df_freq[:num], x='Palavra', y='Frequencia',
                            title=f"Distribuição de Frequências das Palavras Normalizadas ({num}/{df_freq.shape[0]})",
                            labels={'Frequencia': 'Frequência'},
                            text='Frequencia')
            st.write(fig_freq)
            st.write(df_freq[:num])
        
        with tab3:
            num = st.slider("Escolha a quantidade de palavras que serão exibidas", 0, 100, 20, key="s1")
            st.write("Palavras normalizadas com maior peso para avaliações positivas")
            st.write(pesos.nlargest(num, 0))
            st.write("Palavras normalizadas com maior peso para avaliações negativas")
            st.write(pesos.nsmallest(num, 0))


if __name__ == "__main__":
    modelo, vetor = load_model()
    tokenizer = WordToken()    
    streamlit_page(modelo, vetor, tokenizer)

#streamlit run NPL/6-streamlit.py
