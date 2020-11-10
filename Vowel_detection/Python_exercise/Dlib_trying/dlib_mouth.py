__author__ = "Luke Liu"
#encoding="utf-8"
import  dlib
import cv2
import numpy as np
import os


def get_rectangle(landmarks):
    x_list=[]
    y_list=[]
    for idx, point in enumerate(landmarks):
        x_list.append(point[0,0])
        y_list.append(point[0,1])

    x_min=min(x_list)
    x_max=max(x_list)
    y_min = min(y_list)
    y_max = max(y_list)

    return (x_min,x_max,y_min,y_max)


def extract_mouth_roi(lib_path):
    '''
    lib_path: dlib libraries
    '''

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(lib_path)

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if frame is not None:
            img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            # Face regions
            rects = detector(img_gray, 0)
            nums_faces = len(rects)
            for (i, rect) in enumerate(rects):
                # 64 key points
                landmarks = np.matrix([[p.x, p.y] for p in predictor(frame, rect).parts()])
                (x_min, y_min, x_max, y_max) = get_rectangle(landmarks[48:])
                for idx, point in enumerate(landmarks):
                    if idx > 47:
                        pos = (point[0, 0], point[0, 1])
                        x = point[0, 0]
                        y = point[0, 1]

                        cv2.circle(frame, pos, 2, color=(0, 255, 0))
                        cv2.rectangle(frame, (x_min - 10, y_min - 10, x_max - x_min + 15, y_max - y_min + 15),
                                      (0, 255, 0), 2)
                        cv2.imshow("", frame)

            k = cv2.waitKey(1)
            if k == 27:
                break

    cap.release()

if __name__ == '__main__':
    extract_mouth_roi(r"dlibmodels/shape_predictor_68_face_landmarks.dat")
