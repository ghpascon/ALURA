import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from scipy import ndimage
import pickle

def plot_img(img,labels, img_range=1, labels_map=None):
    plt.figure()

    for image in range(img_range):
        plt.subplot(2, int(img_range/2), image+1)
        plt.imshow(img[image])
        plt.axis('off')
        plt.title(f"{labels_map[labels[image]]}")

    plt.tight_layout()    
    plt.show()

def img_treatments(x_train, x_test, px_max_val):
    return x_train/px_max_val, x_test/px_max_val

def plot_model_hist(model_hist):
    loss = model_hist.history.get('loss')
    val_loss = model_hist.history.get('val_loss')
    accuracy = model_hist.history.get('accuracy')
    val_accuracy = model_hist.history.get('val_accuracy')

    # Subplots para loss e accuracy
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plotar a perda de treino e validação
    axs[0].plot(loss, label='Train Loss')
    axs[0].plot(val_loss, label='Validation Loss')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()

    # Plotar a acurácia de treino e validação
    axs[1].plot(accuracy, label='Train Accuracy')
    axs[1].plot(val_accuracy, label='Validation Accuracy')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Exibir os plots
    plt.tight_layout()
    plt.show()

def save_obj(obj, name):
    with open(name, 'wb') as arquivo:
        pickle.dump(obj, arquivo)    
        print(f'Objeto salvo em: {name}')

if __name__ == '__main__':
    dataset = keras.datasets.fashion_mnist
    ((x_train, y_train),(x_test, y_test)) = dataset.load_data()
    print(f'Train: {len(x_train)}, Test: {len(x_test)}')
    
    total_de_classificacoes = len(np.unique(y_train))
    print(f'Classificacoes: {total_de_classificacoes}')

    labels_map = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Boot"
    }

    # plot_img(
        # img= x_train,
        # labels=y_train,
        # img_range = 10,
        # labels_map=labels_map
        # )

    px_max_val = 255
    x_train, x_test = img_treatments(x_train, x_test, px_max_val)

    '''
    Relu -> aplicar funções não lineares
    Softmax -> define uma porcentagem para cada classe
    '''
    units = 128
    dropout = 0.2
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(x_train.shape[1], x_train.shape[2])),
        keras.layers.Dense(units, activation=tensorflow.nn.relu),
        keras.layers.Dropout(dropout),
        keras.layers.Dense(total_de_classificacoes, activation=tensorflow.nn.softmax)
    ])

    #compilar o modelo
    optmizer='adam'
    loss='sparse_categorical_crossentropy'
    metrics = ['accuracy']
    model.compile(optimizer=optmizer, loss=loss, metrics=metrics)
    
    #treinar o modelo
    epochs = 9
    validation_split=0.2
    model_hist=model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split, verbose='auto', batch_size=epochs)
    plot_model_hist(model_hist)

    #testing
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')

    save_obj(model, 'keras/model/model.pkl')
    save_obj(x_test, 'keras/model/x_test.pkl')
    save_obj(y_test, 'keras/model/y_test.pkl')
    save_obj(labels_map, 'keras/model/labels_map.pkl')

    img_dimensions = (x_train.shape[1], x_train.shape[2])
    save_obj(img_dimensions, 'keras/model/img_dimensions.pkl')


