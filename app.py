import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from PIL import Image
from tempfile import NamedTemporaryFile
from keras.utils import load_img, img_to_array
import io


st.title('Flower Image Recognition')
image_file = st.file_uploader('Upload flower image  here')

# image = Image.open(image_file)
# st.image(image)

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

        if (prediction[0][0] > prediction[0][1]) & (prediction[0][0] > prediction[0][2]):
            prediction_daisy = np.round(prediction[0][0]*100, decimals=1)
            st.write('This flower is daisy -->', str(prediction_daisy), '%')

        elif (prediction[0][1] > prediction[0][0]) & (prediction[0][1] > prediction[0][2]):
            prediction_sunflower = np.round(prediction[0][1]*100, decimals=1)
            st.write('This flower is sunflower -->', str(prediction_sunflower), '%')

        elif (prediction[0][2] > prediction[0][0]) & (prediction[0][2] > prediction[0][1]):
            prediction_tulip = np.round(prediction[0][2]*100, decimals=1)
            st.write('This flower is tulip -->', str(prediction_tulip), '%')

    else:
        st.write('')
