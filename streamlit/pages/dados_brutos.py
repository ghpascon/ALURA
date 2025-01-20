import streamlit as st
import requests
import pandas as pd
import time

def get_csv_file(dados):
    return dados.to_csv(index = False).encode('utf-8')

st.title('DADOS BRUTOS')

def get_products():
    status_code = 0
    while status_code != 200:
        url = "https://labdados.com/produtos"
        response = requests.get(url)
        status_code = response.status_code
        if status_code!= 200:
            sleep(500)
    dados = pd.DataFrame.from_dict(response.json())
    return dados

def menssagem_sucesso():
    sucesso = st.success('Arquivo baixado com sucesso!', icon = "✅")
    time.sleep(5)
    sucesso.empty()

dados = get_products()
dados['Data da Compra'] = pd.to_datetime(dados['Data da Compra'], format = '%d/%m/%Y')

with st.expander('Colunas'):
    colunas = st.multiselect('Selecione as colunas:', list(dados.columns), list(dados.columns))

st.sidebar.title('Filtros')
with st.sidebar.expander('Nome do produto'):
    produtos = st.multiselect('Selecione os produtos', dados['Produto'].unique(), dados['Produto'].unique())
with st.sidebar.expander('Preço do produto'):
    preco = st.slider('Selecione o preço', 0, 5000, (0,5000))
with st.sidebar.expander('Data da compra'):
    data_compra = st.date_input('Selecione a data', (dados['Data da Compra'].min(), dados['Data da Compra'].max()))
    data_compra = pd.to_datetime(data_compra)

filtered_dados = dados[(dados['Produto'].isin(produtos)) & (dados['Preço'] >= preco[0]) & (dados['Preço'] <= preco[1]) & (dados['Data da Compra'] >= data_compra[0]) & (dados['Data da Compra'] <= data_compra[1])] 

st.download_button('Download CSV', get_csv_file(filtered_dados[colunas]), 'dados', mime = 'text/csv', on_click=menssagem_sucesso)
st.dataframe(filtered_dados[colunas])

