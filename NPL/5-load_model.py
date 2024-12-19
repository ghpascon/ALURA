import pandas as pd
from nltk import tokenize, FreqDist, corpus, RSLPStemmer, download
from unidecode import unidecode
import pickle
import warnings

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

if __name__ == "__main__":
    modelo, vetor = load_model()
    tokenizer = WordToken()

    while True:
        # Capture user opinion
        input_data = input("\nColoque sua opinião: ")

        if not input_data.strip():
            print("Opinião vazia! Tente novamente.")
            continue

        # Process opinion for tokenization and treatment
        _, processed_data = tokenizer.word_tokenization(input_data)

        # Transform the processed data with the vectorizer
        input_vectorized = vetor.transform(processed_data)

        # Make the prediction
        prediction = modelo.predict(input_vectorized)
        probabilities = modelo.predict_proba(input_vectorized)

        # Display the results
        print(f"\nOpinião: '{input_data}' => Previsão: {prediction[0]} com {max(probabilities[0])*100:.2f}%")
        print(f"Probabilidade de cada classe: {probabilities[0]}")

        # Analyze word impact
        print("\nImpacto das palavras:")
        feature_names = vetor.get_feature_names_out()
        input_words = processed_data[0].split()  # Words in the processed opinion

        for word in input_words:
            if word in feature_names:
                index = list(feature_names).index(word)
                weight = modelo.coef_[0][index]
                print(f"Palavra: '{word}' => Peso: {weight:.4f}")
            else:
                print(f"Palavra: '{word}' => Não encontrada no vocabulário treinado.")
