import spacy.displacy
import streamlit as st
import spacy
from spacy_streamlit import visualize_ner

model = spacy.load('modelo')

rotulos = list(model.get_pipe('ner').labels)
cores = {
    'B-JURISPRUDENCIA':'#0000ff',
    'B-LEGISLACAO':"#ff0000",
    'B-LOCAL':'#00ff00',
    'B-ORGANIZACAO':"#ffff00",
    'B-PESSOA':"#ffffff",
    'B-TEMPO':'#00ffff',
    'I-JURISPRUDENCIA':'#0000ff',
    'I-LEGISLACAO':"#ff0000",
    'I-LOCAL':'#00ff00',
    'I-ORGANIZACAO':"#ffff00",
    'I-PESSOA':"#ffffff",
    'I-TEMPO':'#00ffff',
    'LOC':'#cccccc',
    'MISC':'#cccccc',
    'ORG':'#cccccc',
    'PER':'#cccccc'
}
opcoes = {'ents':rotulos,'colors':cores}
st.title('Reconhecimento de entidades (NER)')

escolha = st.radio('Escolha uma opção: ',options=['Texto','Arquivo'])
texto = None
if escolha == 'Texto':
    texto = st.text_area('Insira o texto')

elif escolha == 'Arquivo':
    arquivo = st.file_uploader('Faça o upload do arquivo para analize', type='txt')
    if arquivo:
        texto = arquivo.read().decode('utf-8')
if texto is not None:
    doc = model(texto)
    visualize_ner(doc, labels=rotulos, displacy_options=opcoes,title='Reconhecimento de Entidades')