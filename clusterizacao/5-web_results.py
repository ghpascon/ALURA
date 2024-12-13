import pickle
import os
import  streamlit as st
import pandas as pd


def load_obj(filename):
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # nome_arquivo = os.path.join(script_dir, filename)
    try:
        with open(filename, 'rb') as arquivo:
            return pickle.load(arquivo)

    except Exception as e:
        print(f"Erro ao carregar o arquivo: {e}")   

def processar_prever (df, model):
    encoded_sexo = encoder.transform(df[['sexo']])
    encoded_df = pd.DataFrame(encoded_sexo, columns=encoder.get_feature_names_out(['sexo']))
    dados = pd.concat([df.drop('sexo', axis=1), encoded_df], axis=1)
        
    dados_escalados = scaler.transform(dados)

    cluster = model.predict(dados_escalados)

    return cluster

if __name__ == "__main__":
    model = load_obj('obj/model.pkl')
    encoder = load_obj('obj/encoder.pkl')
    scaler = load_obj('obj/scaler.pkl')

    st.title('Clusterização para Marketing')
    st.write('Aplicação de KMeans para clusterizar os dados em diversos grupos')

    up_file = st.file_uploader('Upload csv para previsão', type='csv')

    if up_file is not None:
        st.write("""
                        ### Descrição dos Grupos:
                        - **Grupo 0** é focado em um público jovem com forte interesse em moda, música e aparência.
                        - **Grupo 1** está muito associado a esportes, especialmente futebol americano, basquete e atividades culturais como banda e rock.
                        - **Grupo 2** é mais equilibrado, com interesses em música, dança, e moda.
                    """)
        df = pd.read_csv(up_file)
        cluster = processar_prever(df, model)
        df.insert(0, 'grupos', cluster)
        
        st.write('Visualização dos resultados (10 primeiros registros):')
        st.write(df.head(10))
        
        csv = df.to_csv(index=False)
        st.download_button(label='Baixar resultados completos', data = csv, file_name = 'Grupos_interesse.csv', mime='text/csv')
    
#streamlit run clusterizacao/5-web_results.py