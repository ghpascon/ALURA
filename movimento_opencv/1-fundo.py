import cv2
import numpy as np
from time import sleep

video = cv2.VideoCapture(r'movimento_opencv\videos\Rua.mp4')
delay = 10
captura_ok, frame = video.read()

frame_ids = video.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=72)
frames = []
for fid in frame_ids:
    video.set(cv2.CAP_PROP_POS_FRAMES, fid)
    captura_ok, frame = video.read()
    # frames.append(frame)
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)

while (True):
    tempo = float(1/delay)
    sleep(tempo)
    captura_ok, frame = video.read()

    if not captura_ok:
        print('Acabou os frames')
        break

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dframe = cv2.absdiff(frameGray, median_frame)
    th, dframe = cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    cv2.imshow('Frames em Cinza', dframe)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()