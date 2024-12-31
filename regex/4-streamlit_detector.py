import pandas as pd
import re

from nltk.util import bigrams, everygrams
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.tokenize import WhitespaceTokenizer
from nltk.lm import MLE, NgramCounter, Laplace

from sklearn.model_selection import train_test_split

import pickle

import streamlit as st
import plotly.express as px

def substituir_regex(data, regex, substituir = ''):
    if type(data) == str:
        return regex.sub(substituir,data)
    return [regex.sub(substituir, dat) for dat in data]

def regex_data(data):
    regex_punctuation = re.compile(r'[^\w\s]') # Remove as pontuações
    regex_digit = re.compile(r'\d+') # Remove dígitos
    regex_line = re.compile(r'\n') # Remove quebra de linha

    # Realiza substituições sequenciais
    data = substituir_regex(data, regex_punctuation)
    data = substituir_regex(data, regex_digit)
    data = substituir_regex(data, regex_line, ' ')

    # Remove espaços duplicados ao final
    regex_duplicate_space = re.compile(r'\s+')  # Normaliza múltiplos espaços
    data = substituir_regex(data, regex_duplicate_space, ' ')
    return data

def get_bigrams(data):
    data = regex_data(data)
    tokenizer = WhitespaceTokenizer()
    tokens = tokenizer.tokenize(data.lower())

    fakechar = [list(pad_both_ends(palavra, n=2)) for palavra in tokens]
    return [list(bigrams(palavra)) for palavra in fakechar]

def test_perplexity(texto, model):
    return sum([model.perplexity(txt) for txt in get_bigrams(texto)])

# test_perplexity(pt_test.iloc[0], model_in)

def get_best_perplexity(texto, models):
    perplexity_list = [(test_perplexity(texto, model)) for model in models]
    best_index = perplexity_list.index(min(perplexity_list))
    return perplexity_list, best_index


def streamlit_page(models, model_labels):
    st.set_page_config(
        page_title="Detector de Idioma",
        layout="wide",
        page_icon="NPL/data/alura.png",
        )
    
    st.title("Detector de Idioma.")
    st.write("Coloque uma frase em Português, Inglês ou Espanhol para o Modelo realizar a previsão.")
    
    texto = st.text_input("Texto para previsão:")
    if texto:
        perplexity_list, best_index = get_best_perplexity(texto, models)
        st.write(f'O Idioma do texto é -> {model_labels[best_index]}')

        st.divider()
        st.write('Gráfico de perplexidade para cada Idioma')

        df = pd.DataFrame({
            "Modelo": model_labels.values(),  # Rótulos personalizados
            "Perplexidade": perplexity_list
        })

        # Criar o gráfico de barras
        fig = px.bar(df, x="Modelo", y="Perplexidade", title="Perplexidade por Modelo", labels={"Perplexidade": "Perplexidade"})
        fig.update_layout(xaxis_title="Idiomas", yaxis_title="Perplexidade")
        st.write(fig)


if __name__ == "__main__":
    with open("regex/model/models.pkl", 'rb') as arquivo:
        models = pickle.load(arquivo)
    with open("regex/model/model_labels.pkl", 'rb') as arquivo:
        model_labels = pickle.load(arquivo)        
    print('models lodaded')
    streamlit_page(models, model_labels)

#streamlit run regex/4-streamlit_detector.py
