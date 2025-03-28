from main import db
from textblob import TextBlob
from googletrans import Translator
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd


class AnalizadorSentimento():
    def __init__(self):
        self.translator = Translator()

    def analise(self, frase):
        frase_traduzida = self.translator.translate(frase, dest='en')
        frase_traduzida = str(frase_traduzida)
        tb = TextBlob(frase_traduzida)
        return tb.sentiment.polarity
analizador = AnalizadorSentimento()

class PrevisaoCasa():
    def __init__(self):
        with open("ml_models/model2.pkl", "rb") as file:
            self.model = pickle.load(file)
    def prever(self, tamanho, ano, garagem):
        entrada = pd.DataFrame({
            'tamanho': [int(tamanho)],
            'ano': [int(ano)],
            'garagem': [int(garagem)],
            })
        valor = self.model.predict(entrada)[0] 
        valor_formatado = f"R${valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return valor_formatado
previsao_casa = PrevisaoCasa()