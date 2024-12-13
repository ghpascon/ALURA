from ucimlrepo import fetch_ucirepo
import plotly.express as px

def get_dataframe():
    heart_disease = fetch_ucirepo(id=45)
    
    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # variable information
    print(heart_disease.variables) 

    return X, y

if __name__ == "__main__":  
    X, y = get_dataframe()
    data = X

    # Separar entre sim ou não para chance de doença
    y = (y > 0).astype(int)

    data["doenca"] = y

    print(X.head())
    print(y.head())

    # Plot com Plotly
    fig = px.histogram(
        data,
        x="age",
        nbins=30,
        title="Histogram"
    )
    fig.show(renderer="browser")#provavelmente nao carrega, mas gera de forma correta
