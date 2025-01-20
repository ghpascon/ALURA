import cv2
import mediapipe as mp

mp_maos = mp.solutions.hands
mp_desenho = mp.solutions.drawing_utils
maos = mp_maos.Hands()

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
    for mao in maos_dedos:
        print(mao[0], mao[1])

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


video = cv2.VideoCapture(0)

while True:
    captura_ok, frame = video.read()
    if captura_ok:
        frame = cv2.flip(frame, 1)
        frame, todas_maos = treat_frame(frame)
        maos_dedos = dados_maos(todas_maos)
        acao_maos(maos_dedos)
        cv2.imshow('video', frame)

        # Para encerrar, pressione a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        video.release()
        cv2.destroyAllWindows()