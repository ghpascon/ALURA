import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
from scipy import ndimage
import pickle
from PIL import Image, ImageOps

def load_file(name):
    try:
        with open(name, 'rb') as arquivo:
            return pickle.load(arquivo)

    except FileNotFoundError as e:
            return  f'Erro ao carregar {name}'

def img_treatment(img_path, img_dimensions):
    """
    Trata a imagem de entrada para ser compatível com o modelo treinado.
    """
    try:
        # Abrir a imagem
        img = Image.open(img_path)

        # Converter para escala de cinza (1 canal)
        img = img.convert("L")

        # Redimensionar para 28x28
        img = img.resize(img_dimensions, Image.Resampling.LANCZOS)

        # Converter para numpy array
        img_array = np.array(img, dtype=np.float32)

        # Normalizar (0-1, conforme treinamento)
        img_array = img_array / 255.0

        # # Ajustar para o formato (1, 28, 28, 1)
        # img_array = np.expand_dims(img_array, axis=-1)  # Adiciona o canal
        # img_array = np.expand_dims(img_array, axis=0)   # Adiciona o batch
        # img_array = np.reshape(img_array, (1, img_dimensions[0], img_dimensions[1]))
       
        # Expand dimensions to simulate a batch of 1
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    except Exception as e:
        print(f"Erro ao tratar a imagem: {e}")
        return None
    

def plot_image(image, title="Imagem Tratada"):
    """
    Plota uma imagem usando matplotlib.
    
    Args:
        image (Image): Imagem a ser plotada (Pillow Image).
        title (str): Título do gráfico.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image)  # Usa escala de cinza
    plt.axis('off')  # Remove os eixos
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    model = load_file('keras/model/model.pkl')  
    print(model.summary())

    x_test = load_file('keras/model/x_test.pkl')
    y_test = load_file('keras/model/y_test.pkl')
    labels_map = load_file('keras/model/labels_map.pkl')
    img_dimensions = load_file('keras/model/img_dimensions.pkl')
    
    #testing
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')    

    predictions = model.predict(x_test)
    predictions_class = np.argmax(predictions, axis=1) 

    plot_image(x_test[0])
    print(labels_map[predictions_class[0]])
    plot_image(x_test[1])
    print(labels_map[predictions_class[1]])
    plot_image(x_test[2])
    print(labels_map[predictions_class[2]])
    plot_image(x_test[3])
    print(labels_map[predictions_class[3]])


