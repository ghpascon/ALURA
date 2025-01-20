import cv2
import sys

import cv2.bgsegm  # Certifique-se de ter opencv-contrib-python instalado

def Subtractor(algorithm_type):
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'GMG':
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if algorithm_type == 'CNT':
        return cv2.bgsegm.createBackgroundSubtractorCNT()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    print('Erro - Algoritmo inválido. Insira uma opção válida.')
    sys.exit(1)

# Lista de algoritmos suportados
mascara_algoritimos = ['KNN', 'GMG', 'CNT', 'MOG2']
algoritimo = mascara_algoritimos[0]  # Escolha 'MOG' como padrão

# Carregar o vídeo
video = cv2.VideoCapture(r'movimento_opencv\videos\Ponte.mp4')
if not video.isOpened():
    print("Erro ao carregar o vídeo. Verifique o caminho.")
    sys.exit(1)

# Criar o subtrator de fundo
background_subtractor = Subtractor(algoritimo)

t1 = cv2.getTickCount()
while video.isOpened():
    ok, frame = video.read()
    if not ok:
        print("Fim do vídeo ou erro na leitura do frame.")
        break
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # Aplicar a máscara de fundo
    fg_mask = background_subtractor.apply(frame)

    # Mostrar os resultados
    cv2.imshow('FRAME', frame)
    cv2.imshow('MASK', fg_mask)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

t2 = cv2.getTickCount()
print((t2-t1)/cv2.getTickFrequency())
# Liberar recursos
video.release()
cv2.destroyAllWindows()
