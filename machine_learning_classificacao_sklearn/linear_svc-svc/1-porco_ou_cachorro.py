'''
Classificação binária
Aprendizado supervisionado 

Definir se é porco ou cachorro com base em features
1-pelo longo,2-perna curta,3-auau
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

    dados = porcos + cachorros
    classes = [1] * len(porcos) + [0] * len(cachorros)

    print("Dados:", dados)
    print("Classes:", classes)

    #criando o modelo
    from sklearn.svm import LinearSVC

    model = LinearSVC()
    model.fit(dados, classes)

    #testando o modelo com dados novos
    test_animals = [
        [1,1,1],
        [1,1,0],
        [0,1,1],
    ]

    test_classes = [0,1,0]

    previsao = model.predict(test_animals)
    print(previsao)

    acuracia = (previsao == test_classes).sum() / len(previsao)

    print(f'Acurácia: {acuracia*100}%')    