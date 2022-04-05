# model deployment using all in one method

import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.io import read_file
from tensorflow.image import decode_image, resize

from tensorflow.keras.models import load_model
savedModel=load_model('model_imp.h5')


classes=['airplane', 'automobile', 'bird', 'cat', 'deer', 
        'dog', 'frog', 'horse', 'ship', 'truck']


st.title("Image Prediction")
img_upl = st.file_uploader("Upload an image...", type=["jpg", "jpeg"])


# inference
new_data = Image.open(img_upl)
def load_and_preprocess(filename, img_shape=32):
        img = filename
        img = tf.image.resize(img, size =[img_shape, img_shape])
        img = img/255.
        return img
res = savedModel.predict(tf.expand_dims(load_and_preprocess(np.array(new_data)), axis=0))
res = np.argmax(res)
res = classes[res]
st.title(res)