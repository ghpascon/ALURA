import streamlit as st
from web_app import utils

if __name__ == '__main__':
    if 'model' not in st.session_state:
        st.session_state.model = utils.MODELS()  

    model_class = st.session_state.model
   
    st.set_page_config(
        page_title="Classificador de Noticia",
        layout="wide",
        )
    
    st.title("Classificador de noticías com base no seu título.")
    st.divider()

    st.write('Coloque um título para o modelo realizar a classificação, com base nas seguintes opções:')
    st.write(model_class.classificador.classes_)

    texto = st.text_input("Titulo:")
    if texto:
        progresso = st.progress(0)

        st.write(f'Título escolhido -> {texto}')
        progresso.progress(20)
        
        tokens = model_class.tokenizador(texto)
        progresso.progress(60)
        
        vetor = model_class.combinar_vet_sum(tokens)
        
        # Obter as probabilidades para todas as classes
        probabilidades = model_class.classificador.predict_proba(vetor)

        # Obter a classe com maior probabilidade
        category = model_class.classificador.predict(vetor)

        # Encontrar a probabilidade da classe prevista
        probabilidade = probabilidades[0][model_class.classificador.classes_.tolist().index(category[0])]

        # Exibir a categoria e a probabilidade
        progresso.progress(100)
        st.write(f'A categoria da notícia é: {category[0]} com {probabilidade*100:.2f}% de probabilidade')
#streamlit run word_embeeding/6-web.py