import face_alignment
import collections
import cv2
import numpy as np
from cv_draw import draw_landmarks


cap = cv2.VideoCapture('Lapenko.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('with_landmarks.avi', fourcc, 33.0, (1280, 720))

fa = face_alignment.FaceAlignment(device="cuda:0", flip_input=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("finish")
        break
    input_img = frame
    preds = fa.get_landmarks(input_img)
    if preds:
        for i in range(len(preds)):
            input_img = draw_landmarks(input_img, preds[i])
    out.write(input_img)
    cv2.imshow('frame', input_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
