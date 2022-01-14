import cv2
import streamlit as st
import os
from utils.detection import make_prediction, draw_detection_result


def detect_from_image():
    st.subheader('Detection from image')
    preview = st.image([])
    form = st.form(key='my_form')
    file = form.file_uploader(label='Upload photo', type=['png', 'jpeg', 'jpg'])
    submit = form.form_submit_button('Submit')
    if submit:
        file_url = os.path.join("temp", file.name)
        with open(file_url, "wb") as f:
            f.write(file.getbuffer())
            f.close()
        image = cv2.cvtColor(cv2.imread(file_url), cv2.COLOR_BGR2RGB)
        preview.image(image)
        pred_class, scores, boxes = make_prediction(image)
        image, licences = draw_detection_result(pred_class,scores,boxes,image)
        print(licences)
        preview.image(image)


