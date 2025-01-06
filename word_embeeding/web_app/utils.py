import pickle
from gensim.models import KeyedVectors
import spacy
import numpy as np
import subprocess
import sys
def install_spacy_model():
    try:
        spacy.load("pt_core_news_sm")
        print('NLP já carregado')
    except OSError:
        result = subprocess.run(
            [sys.executable, "-m", "spacy", "download", "pt_core_news_sm"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode == 0:
            print("O pacote 'pt_core_news_sm' foi instalado com sucesso!")
        else:
            print(f"Erro ao instalar o pacote: {result.stderr}")

class MODELS:
    def __init__(self):
        try:
            install_spacy_model()

            self.nlp = spacy.load('pt_core_news_sm', disable=['paser', 'ner', 'tagger', 'textcat'])

            self.w2v_model = KeyedVectors.load_word2vec_format('word_embeeding/model/sg_model.txt')

            with open('word_embeeding/model/lr_model_sg.pkl', 'rb') as f:
                self.classificador = pickle.load(f)
            print('Modelos carregados com sucesso')
        except KeyError:
            print('Erro ao carregar modelos')
            
    def tokenizador(self, texto):
        doc = self.nlp(texto)
        lista = []
        for txt in doc:
            if not txt.is_stop and txt.is_alpha:
                lista.append(txt.text.lower())

        return lista

    def combinar_vet_sum(self, tokens):
        result = np.zeros((1, self.w2v_model.vector_size))
        for token in tokens:
            try:
                result += self.w2v_model.get_vector(token)
            except KeyError:
                pass
        return result

if __name__ == "__main__":
    model_class = MODELS()
    print(model_class.combinar_vet_sum(model_class.tokenizador('Rio de Janeiro é uma cidade maravilhosa')))