import cv2
import mediapipe as mp
import os

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands()

acao_feita = False
texto_digitado = ''

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
    global acao_feita, texto_digitado
    acao = None
    make_action = False
    for mao in maos_dedos:
        if mao[0] == 'Left':
            if mao[1] == [True, True, True, True, True]:
                make_action = True
            elif mao[1] == [False, False, False, False, False]:
                acao_feita = False

        if mao[0] == 'Right':
            if mao[1] == [True, True, True, True, True]:
                acao = 0               


    if make_action and acao is not None and not acao_feita:
        if acao == 0:
            texto_digitado = texto_digitado[:-1]
            acao_feita = True

    return make_action

def pegar_coordenadas(hand_landmarks, resolucao_x, resolucao_y):
    coordenadas = []
    for landmark in hand_landmarks.landmark:
                coord_x = int(landmark.x * resolucao_x)
                coord_y = int(landmark.y * resolucao_y)
                coord_z = int(landmark.z * resolucao_x)  # Assuming z is scaled by resolucao_x
                coordenadas.append((coord_x, coord_y, coord_z))
    return coordenadas

def treat_frame(frame):
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
            mp_desenho.draw_landmarks(frame, hand_landmarks, mp_maos.HAND_CONNECTIONS)

    return frame, todas_maos 


def desenhar_teclado(frame, make_action, todas_maos):
    global acao_feita
    
    cores = [
        (255, 255, 255),  # Branco para o fundo das teclas
        (0, 0, 0),        # Preto para o texto
        (255, 0, 0),      # Outras cores não utilizadas, podem ser usadas para realce
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0)
    ]
    teclas = [
        ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
        ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
        ['Z', 'X', 'C', 'V', 'B', 'N', 'M', ',', '.', ' ']
    ]

    # Posição inicial e tamanho das teclas
    tamanho = 50
    margem = 5  # Margem entre as teclas
    inicio_x = 10  # Posição inicial no eixo X
    inicio_y = 10  # Posição inicial no eixo Y

    tecla_pressionada = None

    for linha in teclas:
        pos_x = inicio_x
        for tecla in linha:
            canto_superior = (pos_x, inicio_y)
            canto_inferior = (pos_x + tamanho, inicio_y + tamanho)

            cv2.rectangle(frame, canto_superior, canto_inferior, cores[0], 5)
            cv2.rectangle(frame, canto_superior, canto_inferior, cores[2], 2)

            text_size = cv2.getTextSize(tecla, cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0]
            text_x = pos_x + (tamanho - text_size[0]) // 2
            text_y = inicio_y + (tamanho + text_size[1]) // 2
            cv2.putText(frame, tecla, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1, cores[1], 5)
            cv2.putText(frame, tecla, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1, cores[0], 2)

            pos_x += tamanho + margem

            if make_action and not acao_feita:
                for mao in todas_maos:
                    if mao['lado'] == 'Right' and canto_superior[0] < mao['coordenadas'][8][0] < canto_inferior[0] and canto_superior[1] < mao['coordenadas'][8][1] < canto_inferior[1]:
                        acao_feita = True
                        tecla_pressionada = tecla
    
        inicio_y += tamanho + margem

    return frame, tecla_pressionada

video = cv2.VideoCapture(0)

while True:
    captura_ok, frame = video.read()
    if captura_ok:
        frame = cv2.flip(frame, 1)
        frame, todas_maos = treat_frame(frame)
        maos_dedos = dados_maos(todas_maos)
        make_action = acao_maos(maos_dedos)

        frame, tecla_pressionada = desenhar_teclado(frame, make_action, todas_maos)
        if tecla_pressionada is not None:
            texto_digitado+=tecla_pressionada
        cv2.putText(frame, texto_digitado, (50, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 5)
        cv2.putText(frame, texto_digitado, (50, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)

        cv2.imshow('video', frame)

        # Para encerrar, pressione a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        video.release()
        cv2.destroyAllWindows()