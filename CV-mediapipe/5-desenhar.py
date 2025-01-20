import cv2
import mediapipe as mp
import numpy as np

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands()
desenho_pontos = []
desenhando = False

def dados_maos(todas_maos):
    maos = []
    for mao in todas_maos:
        dedos = []        
        #polegar
        dedos.append(mao['coordenadas'][1][1] > mao['coordenadas'][2][1] > mao['coordenadas'][3][1] > mao['coordenadas'][4][1]
                    and (mao['coordenadas'][1][0] > mao['coordenadas'][2][0] > mao['coordenadas'][3][0] > mao['coordenadas'][4][0] 
                        or mao['coordenadas'][1][0] < mao['coordenadas'][2][0] < mao['coordenadas'][3][0] < mao['coordenadas'][4][0]))
        #indicador
        dedos.append(mao['coordenadas'][5][1] > mao['coordenadas'][6][1] > mao['coordenadas'][7][1] > mao['coordenadas'][8][1])
        #dedo medio
        dedos.append(mao['coordenadas'][9][1] > mao['coordenadas'][10][1] > mao['coordenadas'][11][1] > mao['coordenadas'][12][1])
        #dedo anelar
        dedos.append(mao['coordenadas'][13][1] > mao['coordenadas'][14][1] > mao['coordenadas'][15][1] > mao['coordenadas'][16][1])
        #dedo mindinho
        dedos.append(mao['coordenadas'][17][1] > mao['coordenadas'][18][1] > mao['coordenadas'][19][1] > mao['coordenadas'][20][1])

        maos.append([mao['lado'], dedos])

    return maos

def acao_maos(maos_dedos):  
    global desenho_pontos
    color = (0,0,0)
    draw_available = False
    for mao in maos_dedos:
        if mao[0] == 'Left':
            if mao[1] == [False, True, False, False, False]:
                color = (255, 0, 0)
            elif mao[1] == [False, True, True, False, False]:
                color = (0, 255, 0)  
            elif mao[1] == [False, True, True, True, False]:
                color = (0, 0, 255)     
            elif mao[1] == [False, True, True, True, True]:
                color = (255, 255, 255)                        
        if mao[0] == 'Right':
            if mao[1] == [False, True, False, False, False]:
                draw_available = True
            else: draw_available = False
            if mao[1] == [True, True, True, True, True]:
                desenho_pontos = []
                
        
    return color, draw_available                        

def pegar_coordenadas(hand_landmarks, resolucao_x, resolucao_y):
    coordenadas = []
    for landmark in hand_landmarks.landmark:
                coord_x = int(landmark.x * resolucao_x)
                coord_y = int(landmark.y * resolucao_y)
                coord_z = int(landmark.z * resolucao_x)  # Assuming z is scaled by resolucao_x
                coordenadas.append((coord_x, coord_y, coord_z))
    return coordenadas

def treat_frame(frame, frame_branco):
    # Get the frame resolution
    resolucao_y, resolucao_x, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = maos.process(rgb_frame)

    todas_maos = []
    if results.multi_hand_landmarks:
        for lado_mao, hand_landmarks in zip(results.multi_handedness, results.multi_hand_landmarks):
            coordenadas=pegar_coordenadas(hand_landmarks, resolucao_x, resolucao_y)
            lado = lado_mao.classification[0].label

            info_maos = {'coordenadas': coordenadas,
                         'lado': lado}
            
            todas_maos.append(info_maos)
            mp_desenho.draw_landmarks(frame_branco, hand_landmarks, mp_maos.HAND_CONNECTIONS)

    return frame_branco, todas_maos 

def desenhar_frame(frame, draw_color, draw_available, todas_maos):
    global desenho_pontos, desenhando

    coordenadas_indicador = (0, 0)
    espessura=None
    for mao in todas_maos:
        if mao['lado'] == 'Right':
            coordenadas_indicador = (mao['coordenadas'][8][0], mao['coordenadas'][8][1])
            espessura = int(abs(mao['coordenadas'][8][2])) // 3 + 5

    if draw_available:
        if not desenhando:
            # Adiciona um ponto de quebra lógico
            desenho_pontos.append(None)
            desenhando = True
        desenho_pontos.append([coordenadas_indicador, espessura, draw_color])
    else:
        desenhando = False

    # Desenhar linhas, ignorando pontos de quebra (None)
    if len(desenho_pontos) > 1:
        for i in range(len(desenho_pontos) - 1):
            if desenho_pontos[i] is not None and desenho_pontos[i + 1] is not None:
                cv2.line(
                    frame,
                    desenho_pontos[i][0],
                    desenho_pontos[i + 1][0],
                    desenho_pontos[i][2],
                    desenho_pontos[i][1],
                )

    if espessura is not None:
        cv2.circle(frame, coordenadas_indicador, espessura, draw_color, cv2.FILLED)

    return frame
   
    

video = cv2.VideoCapture(0)

while True:
    captura_ok, frame = video.read()
    if captura_ok:
        frame = cv2.flip(frame, 1)
        frame_branco = np.ones(frame.shape, dtype=np.uint8)*255
        frame, todas_maos = treat_frame(frame, frame_branco)
        maos_dedos = dados_maos(todas_maos)
        draw_color, draw_available = acao_maos(maos_dedos)
        desenhar_frame(frame, draw_color, draw_available, todas_maos)
        cv2.imshow('video', frame)

        # Para encerrar, pressione a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        video.release()
        cv2.destroyAllWindows()