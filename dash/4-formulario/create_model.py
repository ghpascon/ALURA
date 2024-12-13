from ucimlrepo import fetch_ucirepo
import plotly.express as px
from dash import Dash, dcc, html
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score 

def get_dataframe():
    heart_disease = fetch_ucirepo(id=45)
    
    # Data (as pandas DataFrames)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Variable information
    print(heart_disease.variables)

    return X, y

def get_model_data(data, y_label):
    if data.empty:
        print("Os dados estão vazios.")
        return None,None
    
    x = data.drop(columns=[y_label])
    y = data[y_label]
    
    return x, y

def save_model(model, model_dir_name, filename):
    import pickle
    import os
    
    #salvar o modelo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir, model_dir_name)
    nome_arquivo = os.path.join(model_dir, f"{filename}.pkl")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # Cria o diretório "modelos"

    # Salvar o modelo no arquivo
    with open(nome_arquivo, 'wb') as arquivo:
        pickle.dump(model, arquivo)

    print(f"Modelo salvo em: {nome_arquivo}")    

if __name__ == "__main__":
    data, y = get_dataframe()

    # Separate between 'yes' or 'no' for the chance of disease
    y = (y > 0).astype(int)

    data["doenca"] = y

    print(data.head())


    target = "doenca"
    x, y = get_model_data(data, target)
    print(x.head())
    print(y.head())

    seed = 12
    test_size=0.2
    raw_train_x,raw_test_x,train_y,test_y = train_test_split(x, y,test_size=test_size, random_state=seed, stratify=y)

    model = xgb.XGBClassifier(objective = 'binary:logistic')
    model.fit(raw_train_x, train_y)

    predictions = model.predict(raw_test_x)
    accuracy = accuracy_score(test_y, predictions)
    print(f'Acuracy: {accuracy*100}%')

    model_dir="models"
    filename="model_xgboost"
    save_model(model, model_dir, filename)



