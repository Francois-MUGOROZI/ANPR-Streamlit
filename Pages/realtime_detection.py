import cv2
import streamlit as st
import os
from utils.detection import make_prediction, draw_detection_result
import time


def detect_realtime():
    cap = cv2.VideoCapture(0)
    st.subheader('Realtime detection')
    preview = st.image([])
    while True:
        try:
            _, frame = cap.read()
            preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pred_class, scores, boxes = make_prediction(frame)
            image, licence = draw_detection_result(pred_class, scores, boxes, frame)
            preview.image(image)
            time.sleep(5)
        except:
            pass
    cap.release()
    cv2.destroyAllWindows()