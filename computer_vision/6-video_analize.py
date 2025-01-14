import cv2 as cv
import numpy as np
import dlib
import matplotlib.pyplot as plt
from scipy.spatial import distance 

classificado_dlib_68 = dlib.shape_predictor("computer vision/classificador/shape_predictor_68_face_landmarks.dat")
face_detector = dlib.get_frontal_face_detector()

FACE = list(range(17, 68))
FACE_COMPLETA = list(range(0, 68))
LABIO = list(range(48, 61))
SOMBRANCELHA_DIRETA = list(range(17, 22))
SOMBRANCELHA_ESQUERDA = list(range(22, 27))
OLHO_DIREITO = list(range(36, 42))
OLHO_ESQUERDO = list(range(42, 48))
NARIZ = list(range(27, 35))
MANDIBULA = list(range(0, 17))

def get_aspects(macros):
    def aspecto_razao_olhos(pontos_olhos):
        # Verificar a entrada e convertê-la para numpy array se necessário
        pontos_olhos = np.array(pontos_olhos)

        # Calculando as distâncias
        a = distance.euclidean(pontos_olhos[1], pontos_olhos[5])  # Distância entre pontos [1] e [5]
        b = distance.euclidean(pontos_olhos[2], pontos_olhos[4])  # Distância entre pontos [2] e [4]
        c = distance.euclidean(pontos_olhos[0], pontos_olhos[3])  # Distância entre pontos [0] e [3]

        # Calculando a razão de aspecto
        aspecto_razao = (a + b) / (2.0 * c)

        return aspecto_razao
        
    def aspecto_razao_boca(pontos_boca):
        # Verificar a entrada e convertê-la para numpy array se necessário
        pontos_boca = np.array(pontos_boca)

        # Calculando as distâncias
        a = distance.euclidean(pontos_boca[3], pontos_boca[9])  # Distância entre pontos [1] e [5]
        b = distance.euclidean(pontos_boca[2], pontos_boca[10])  # Distância entre pontos [2] e [4]
        c = distance.euclidean(pontos_boca[4], pontos_boca[8])  # Distância entre pontos [0] e [3]
        d = distance.euclidean(pontos_boca[0], pontos_boca[6])  # Distância entre pontos [0] e [3]

        # Calculando a razão de aspecto
        aspecto_razao = (a + b + c) / (3.0 * d)

        return aspecto_razao

    if macros is None:
        return None
    valor_olho_esquerdo = aspecto_razao_olhos(macros[0][OLHO_ESQUERDO])
    valor_olho_direito = aspecto_razao_olhos(macros[0][OLHO_DIREITO])
    valor_boca = aspecto_razao_boca(macros[0][LABIO])
    return [round(valor_olho_esquerdo, 3), round(valor_olho_direito, 3), round(valor_boca, 3)]

def anotar_marcos_casca_convexa(imagem, marcos, ranges, color=(255, 0, 0)): 
    for marco in marcos:
        for range in ranges:
            pontos = cv.convexHull(marco[range])
            cv.drawContours(imagem, [pontos], 0, color, 2)
        
    return imagem

def treatment_frame(frame, treatment):
    def get_face_rect(img):
        rects = face_detector(img, 1)
        return rects
    
    def get_macros(img):
        rect = face_detector(img, 1)
        if len(rect) == 0:
            return None
        
        macros = []
        for ret in rect:
            macros.append(np.matrix([[p.x, p.y] for p in classificado_dlib_68(img,ret).parts()]))
        return macros
        
    if treatment == 0:
        return None, None
    
    if treatment == 1:
        rects = get_face_rect(frame)
        return rects, None
    
    if treatment == 2 or treatment == 3:
        macros = get_macros(frame)
        return None, macros
        


def show_video(path, name='Video', treatment = 0, interval = 10, size=(500, 400)):
    """
    Tratamento:
    0 - sem tratamento
    1 - face detection
    2 - macro detection
    3 - convex hull
    """
    video = cv.VideoCapture(path)
    try:
        
        frame_count = 0
        color = (0, 255, 0)
        rect, macros = None, None
        valores = None

        while True:
            captura_ok, frame = video.read()
            if captura_ok:
                frame = cv.resize(frame, size)                
                frame_count += 1
                if frame_count % interval == 0:
                    rect, macros= treatment_frame(frame, treatment)
                    valores = get_aspects(macros)

                if rect is not None:
                    for k,d in enumerate(rect):
                        cv.rectangle(frame,(d.left(),d.top()), (d.right(),d.bottom()), color, 5)

                if macros is not None and treatment==2:
                    for macro in macros:
                        for idx, ponto in enumerate(macro):
                            centro = (ponto[0, 0], ponto[0, 1])
                            cv.circle(frame, centro, 5, color, -1)   

                if macros is not None and treatment==3:
                    frame = anotar_marcos_casca_convexa(frame, macros, [LABIO, OLHO_ESQUERDO, OLHO_DIREITO])

                if valores is not None:
                    texto_olhos = f'Olho esquerdo: {valores[0]} | Olho direito: {valores[1]}'
                    texto_boca = f'Boca: {valores[2]}'
                    cv.putText(frame, texto_olhos, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv.putText(frame, texto_boca, (20, 60), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if valores[0] < 0.270 and valores[1] < 0.270 and valores[2] > 0.500:
                        texto_bocejo = 'BOCEJO DETECTADO!!!'
                        cv.putText(frame, texto_bocejo, (20, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                cv.imshow(name, frame)

                # Para encerrar, pressione a tecla 'q'
                if cv.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                video.release()
                cv.destroyAllWindows()
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
    finally:
        # Libera os recursos da câmera e fecha as janelas
        video.release()
        cv.destroyAllWindows()
        print("Recursos liberados.")

if __name__ =="__main__":
    # show_video(0, treatment=3, interval=20)
    show_video('computer vision/data/bocejo.mov', treatment=3, interval=10)