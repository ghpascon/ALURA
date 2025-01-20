import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
facemesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

tempo_identificar = 1.5
tempo_olho_fechado = time.time()
tempo_boca_aberta = time.time()

tempo_piscadas = time.time()
piscou = False
piscadas = 0

def get_face(frame, draw = True):
    resolucao_y, resolucao_x, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = facemesh.process(rgb_frame)
    coordenadas = []
    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            coordenadas=pegar_coordenadas(face_landmark, resolucao_x, resolucao_y)
            if draw: 
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmark,
                    mp_face_mesh.FACEMESH_CONTOURS,  # Specify the type of connections (e.g., tesselation)
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(102, 255, 102), thickness=1, circle_radius=1),  # Customize connections
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 102, 102), thickness=1, circle_radius=1)      # Customize landmarks
                )
            
    return frame, coordenadas

def pegar_coordenadas(landmarks, resolucao_x, resolucao_y):
    coordenadas = []
    for landmark in landmarks.landmark:
                coord_x = int(landmark.x * resolucao_x)
                coord_y = int(landmark.y * resolucao_y)
                coord_z = int(landmark.z * resolucao_x)  # Assuming z is scaled by resolucao_x
                coordenadas.append((coord_x, coord_y, coord_z))
    return coordenadas


def acao_coordenadas(frame, coordenadas):
    global tempo_olho_fechado, tempo_boca_aberta, piscadas, piscou, tempo_piscadas
    if len(coordenadas) == 0:return frame
    def distancia(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

    def get_olhos():
        p_olho_dir = [385, 380, 387, 373, 362, 263]
        p_olho_esq = [160, 144, 158, 153, 33, 133]
        p_olhos = p_olho_esq + p_olho_dir

        for olho in p_olhos:
            cv2.circle(frame, (coordenadas[olho][0], coordenadas[olho][1]), 3, (0, 0, 255), cv2.FILLED)
        de_1 = distancia(coordenadas[p_olho_esq[0]], coordenadas[p_olho_esq[1]])
        de_2 = distancia(coordenadas[p_olho_esq[2]], coordenadas[p_olho_esq[3]])
        de_3 = distancia(coordenadas[p_olho_esq[4]], coordenadas[p_olho_esq[5]])

        dd_1 = distancia(coordenadas[p_olho_dir[0]], coordenadas[p_olho_dir[1]])
        dd_2 = distancia(coordenadas[p_olho_dir[2]], coordenadas[p_olho_dir[3]])
        dd_3 = distancia(coordenadas[p_olho_dir[4]], coordenadas[p_olho_dir[5]])

        medida_olho_esquerdo = (de_1 + de_2) / (2*de_3)
        medida_olho_direito = (dd_1 + dd_2) / (2*dd_3)
        cv2.putText(frame, f'Olho esquerdo: {medida_olho_esquerdo:.2f}',(30,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f'Olho direito: {medida_olho_direito:.2f}',(30,70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 3)

        return medida_olho_esquerdo, medida_olho_direito
    
    def get_boca():
        p_boca = [82, 87, 13, 14, 312, 317, 78, 308]
        for ponto in p_boca:
            cv2.circle(frame, (coordenadas[ponto][0], coordenadas[ponto][1]), 3, (0, 0, 255), cv2.FILLED)
        d1 = distancia(coordenadas[p_boca[0]], coordenadas[p_boca[1]])
        d2 = distancia(coordenadas[p_boca[2]], coordenadas[p_boca[3]])
        d3 = distancia(coordenadas[p_boca[4]], coordenadas[p_boca[5]])
        d4 = distancia(coordenadas[p_boca[6]], coordenadas[p_boca[7]])
        medida_boca = (d1+d2+d3)/(2*d4)
        cv2.putText(frame, f'Boca: {medida_boca:.2f}',(30,105), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 3)
        
        return medida_boca

    #olhos
    medida_olho_esquerdo, medida_olho_direito = get_olhos() 
    medida_olho_aberto = 0.24
    olho_esquerdo = medida_olho_esquerdo > medida_olho_aberto
    olho_direito = medida_olho_direito > medida_olho_aberto

    cv2.putText(frame, f'Piscadas nos ultimos 60s: {piscadas}',(30,145), cv2.FONT_HERSHEY_COMPLEX, 1, (0,10,10), 3)

    olhos_fechados = not(olho_esquerdo or olho_direito)
    if not olhos_fechados:tempo_olho_fechado = time.time()
    if time.time() - tempo_olho_fechado > tempo_identificar:
        if not piscou:piscadas += 1
        piscou = True
        cv2.putText(frame, 'OLHOS FECHADOS',(30,180), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)
    else:piscou = False
    #boca
    medida_boca = get_boca()
    medida_boca_aberta = 0.8
    boca_aberta = medida_boca > medida_boca_aberta
    if not boca_aberta:tempo_boca_aberta = time.time()
    if time.time() - tempo_boca_aberta > tempo_identificar:
        cv2.putText(frame, 'BOCA ABERTA',(30,220), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 3)

    #tempo piscadas
    if time.time() - tempo_piscadas > 60.0:
        tempo_piscadas = time.time()
        piscadas = 0

    #piscadas
    if piscadas > 6:
        cv2.putText(frame, 'FREQUENCIA ALTA DE PISCADAS NO ULTIMO MINUTO',(30,260), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,255,255), 3)


    #bocejo
    if piscou and boca_aberta:
        cv2.putText(frame, 'BOCEJO DETECTADO',(30,300), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,255), 3)
    
    return frame

     

video = cv2.VideoCapture(0)

while True:
    captura_ok, frame = video.read()
    if captura_ok:
        frame = cv2.flip(frame, 1)

        frame, coordenadas = get_face(frame, False)
        frame = acao_coordenadas(frame, coordenadas)

        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        video.release()
        cv2.destroyAllWindows()