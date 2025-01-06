import pickle
from gensim.models import KeyedVectors
import spacy
import numpy as np


class MODELS:
    def __init__(self):
        try:
            self.nlp = spacy.load('pt_core_news_sm')

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