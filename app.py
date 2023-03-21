import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image
from keras.utils import load_img, img_to_array
import io
import time
# from ultralytics import YOLO


st.title('Flower Image Recognition')
st.caption('Made by Wahyu Dwi Nugraha')

name_input = st.empty()
test = name_input.text_input('Enter your name here')
if test != "":
    name_input.empty()
    st.info(test)
    st.write('Hello ', str(test), ' !!', 'This is a flower recognition system which can predict 3 types of flowers, there are daisy, sunflower, and tulip.')
    st.write('Let"s go predict your flower :)')


    image_file = st.file_uploader('Upload flower image  here')


    if image_file is not None:
        path = image_file.getvalue()
        st.image(path)
        open_image = Image.open(io.BytesIO(path))
        img = tf.image.resize(open_image, [200,200])
        img_array = img_to_array(img)
        img_array = img_array/255

        img_array = img_array.reshape(1,200,200,3)

        if st.button('Predict'):
            model = load_model('Flower_model.h5')

            prediction = model.predict(img_array)

            with st.spinner('Classifying ...'):
                time.sleep(3)

            if (prediction[0][0] > prediction[0][1]) & (prediction[0][0] > prediction[0][2]):
                prediction_daisy = np.round(prediction[0][0]*100, decimals=1)
                st.write('This flower is ', str(prediction_daisy), '%', ' daisy')


            elif (prediction[0][1] > prediction[0][0]) & (prediction[0][1] > prediction[0][2]):
                prediction_sunflower = np.round(prediction[0][1]*100, decimals=1)
                st.write('This flower is ', str(prediction_sunflower), '%', ' sunflower')


            elif (prediction[0][2] > prediction[0][0]) & (prediction[0][2] > prediction[0][1]):
                prediction_tulip = np.round(prediction[0][2]*100, decimals=1)
                st.write('This flower is ', str(prediction_tulip), '%', ' tulip')


        else:
            st.write('')


# This is model for object detection using YOLO
#     from numpy import asarray


#     if image_file is not None:
#         path = image_file.getvalue()
#         st.image(path)
#         open_image = Image.open(io.BytesIO(path))
#         img = tf.image.resize(open_image, [640,640])
#         img_array = asarray(img)
#         # img_array = img_array/255

#         # img_array = img_array.reshape(1,200,200,3)


#         if st.button('Predict'):
#             model = YOLO('Medium_96_ACC.onnx')

#             prediction = model.predict(img_array)

#             with st.spinner('Classifying ...'):
#                 time.sleep(3)
#             print(prediction[0].plot().shape)
#             print(type(prediction[0].plot()))
#             st.image(prediction[0].plot()/255, clamp=True)


#         else:
#             st.write('')
