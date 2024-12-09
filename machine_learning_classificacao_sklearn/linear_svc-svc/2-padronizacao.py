'''
train_x -> dados de treino
train_y -> classes dos dados de treino

test_x -> dados de teste
test_y -> classes dos dados de teste
'''

if __name__ == '__main__':
    #preparando dados
    porcos=[
        [0,1,0],
        [0,1,1],
        [1,1,0],
        ]

    cachorros=[
        [0,1,1],
        [1,0,1],
        [1,1,1],
    ]

    train_x = porcos + cachorros
    train_y = [1] * len(porcos) + [0] * len(cachorros)

    print("Dados:", train_x)
    print("Classes:", train_y)

    #criando o modelo
    from sklearn.svm import LinearSVC

    model = LinearSVC()
    model.fit(train_x, train_y)

    #testando o modelo com dados novos
    test_x = [
        [1,1,1],
        [1,1,0],
        [0,1,1],
    ]

    test_y = [0,1,0]

    previsao = model.predict(test_x)
    print(previsao)

    from sklearn.metrics import accuracy_score

    acuracy = accuracy_score(test_y, previsao)

    print(f'Acurácia: {acuracy*100}%')    