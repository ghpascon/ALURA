import cv2
import sys

def Subtractor(algorithm_type):
    if algorithm_type == 'KNN':
        return cv2.createBackgroundSubtractorKNN()
    if algorithm_type == 'MOG2':
        return cv2.createBackgroundSubtractorMOG2()
    print('Erro - Insira uma nova informação')
    sys.exit(1)

# Lista de algoritmos suportados
mascara_algoritimos = ['KNN', 'MOG2']

for algoritimo in mascara_algoritimos:
    video = cv2.VideoCapture(r'movimento_opencv\videos\Ponte.mp4')
    if not video.isOpened():
        print("Erro ao carregar o vídeo. Verifique o caminho.")
        sys.exit(1)

    # Criar o subtrator de fundo
    background_subtractor = Subtractor(algoritimo)
    frame_cnt = 0
    t1 = cv2.getTickCount()
    while video.isOpened():
        ok, frame = video.read()
        if not ok:
            print("Fim do vídeo ou erro na leitura do frame.")
            break
        frame_cnt+=1
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        # Aplicar a máscara de fundo
        fg_mask = background_subtractor.apply(frame)

        # Mostrar os resultados
        cv2.imshow('FRAME', frame)
        cv2.imshow('MASK', fg_mask)

        # Sair ao pressionar 'q'
        if cv2.waitKey(1) & 0xFF == ord('q') or frame_cnt >500:
            break

    t2 = cv2.getTickCount()
    tempo_processo = (t2-t1)/cv2.getTickFrequency()
    print(f'MASCARA: {algoritimo}, TEMPO: {tempo_processo}')
    # Liberar recursos
    video.release()
    cv2.destroyAllWindows()
