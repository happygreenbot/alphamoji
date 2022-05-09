import numpy as np                   # advanced math library
import matplotlib.pyplot as plt      # MATLAB like plotting routines
import random                        # for generating random numbers

from keras.datasets import mnist     # MNIST dataset is included in Keras
from keras.models import Sequential  # Model type to be used
from keras.models import load_model

from keras.layers.core import Dense, Dropout, Activation # Types of layers to be used in our model
from keras.utils import np_utils                         # NumPy related tools

import cv2
import os

import argparse

import streamlit as st

from config.config import *

import time

import shutil

class AlphaMoji:

    def __init__(self, img_path) -> None:
        # Use the second argument or (flag value) zero
        # that specifies the self.img is to be read in grayscale mode
        self.img = cv2.imread(img_path, 0)
        self.img_resize = None
        self.mat = None
        self.alphamoji = None
        self.model = None

    def print_to_file(self, file_name):
        with open(file_name, 'w') as f:
            for i in range(self.alphamoji.shape[0]):
                for j in range(self.alphamoji.shape[1]):
                    if self.alphamoji[i, j]==5:
                        print(".", file=f, end = "")
                    else:
                        print(self.alphamoji[i, j], file=f, end = "")
                print("", file=f)

    def write_to_string(self):
        alphamoji_str = ""
        for i in range(self.alphamoji.shape[0]):
            for j in range(self.alphamoji.shape[1]):
                if self.alphamoji[i, j]==5:
                    alphamoji_str = alphamoji_str + '.'
                else:
                    alphamoji_str = alphamoji_str + str(self.alphamoji[i, j])
            alphamoji_str = alphamoji_str + '\n'
        return alphamoji_str


    def preserve_aspect_ratio(self):
        target_h = int(self.img.shape[0]/HW_RATIO)
        target_w = self.img.shape[1]
        scale = min(MAX_WIDTH/target_w, 10)
        target_h = int(target_h*scale)
        target_w = int(target_w*scale)
        target_h = round(target_h/INPUT_SIZE)*INPUT_SIZE
        target_w = round(target_w/INPUT_SIZE)*INPUT_SIZE
        print(scale)
        self.img_resize = cv2.resize(self.img, (target_w, target_h))

    def subDivide(self):
        total_l = int(self.img_resize.shape[0]/INPUT_SIZE)
        total_b = int(self.img_resize.shape[1]/INPUT_SIZE)
        print(total_l)
        print(total_b)
        final_array = np.zeros([total_l, total_b, INPUT_SIZE, INPUT_SIZE])
    #     print(final_array.shape)
        for i in range(total_l):
            for j in range(total_b):
                final_array[i, j] = self.img_resize[i*INPUT_SIZE:(i+1)*INPUT_SIZE, j*INPUT_SIZE:(j+1)*INPUT_SIZE]
    #             print(final_array[i, j].shape)
        self.mat = final_array
    #     for i in range(total):


    def transform(self):
        result_dim = [self.mat.shape[0], self.mat.shape[1]]
        print(result_dim)
        result = []
        final_array_reshaped = self.mat.reshape(result_dim[0]*result_dim[1], self.mat.shape[2]*self.mat.shape[3])
        print(final_array_reshaped.shape)
    #     for i in range(final_array_reshaped.shape[0]):
    #         result.append(model.predict_classes(final_array_reshaped[i]))
        print(self.model.predict(final_array_reshaped)[0])
        self.alphamoji = np.array(self.model.predict_classes(final_array_reshaped)).reshape(result_dim)

    def load_mnist_model(self):
        self.model = load_model(MODEL_PATH)

    def return_img(self):
        return self.img

def process_alphamoji_callback(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    input_filepath = "temp/input"
    with open(input_filepath, "wb") as input_file:
        input_file.write(bytes_data)

    obj = AlphaMoji(input_filepath)

    # obj = AlphaMoji(args.input_image)

    preprocess_image_state = st.text('Preprocessing image...')
    # Load 10,000 rows of data into the dataframe.
    obj.preserve_aspect_ratio()

    obj.subDivide()
    # Notify the reader that the data was successfully loaded.
    time.sleep(1)
    preprocess_image_state.text('Preprocessing image...done!')
    
    load_model_state = st.text('Loading model...')
    obj.load_mnist_model()
    time.sleep(1)
    load_model_state.text('Model loaded!')

    alphamoji_state = st.text('Baiting numbers with candy...')
    obj.transform()
    time.sleep(1)
    alphamoji_state.text('Heres your AlphaMoji!')

    st.text(obj.write_to_string())

#     obj.print_to_file("results/chester2_5_large.txt")

    return obj.write_to_string()

if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description='Convert an image into a matrix of numbers and dots')
    # parser.add_argument("input_image", help="path to input image")
    # args = parser.parse_args()

    st.title('AlphaMoji :\'\)')
    st.subheader('Convert your favorite pictures into quirky number matrices! \(inspired by Youtube comment art ofc \)')
    
    if os.path.isdir("temp"):
        shutil.rmtree("temp")
    os.mkdir("temp")
        
    uploaded_file = st.file_uploader("Choose a file")

    if 'output_string' not in st.session_state:
        st.session_state.output_string = None

    if uploaded_file is not None:
        st.session_state.input_image = uploaded_file

    if 'input_image' in st.session_state:
        st.session_state.output_string = process_alphamoji_callback(st.session_state.input_image)

        st.download_button('Download here!', data = st.session_state.output_string, file_name = 'alphamoji.txt')

    # if uploaded_file is not None:
    #     # To read file as bytes:
    #     bytes_data = uploaded_file.getvalue()
    #     input_filepath = "temp/input"
    #     with open(input_filepath, "wb") as input_file:
    #         input_file.write(bytes_data)

    #     obj = AlphaMoji(input_filepath)

    # # obj = AlphaMoji(args.input_image)

    # preprocess_image_state = st.text('Preprocessing image...')
    # # Load 10,000 rows of data into the dataframe.
    # obj.preserve_aspect_ratio()

    # obj.subDivide()
    # # Notify the reader that the data was successfully loaded.
    # preprocess_image_state.text('Preprocessing image...done!')
    
    # load_model_state = st.text('Loading model...')
    # obj.load_mnist_model()
    # load_model_state.text('Model loaded!')

    # alphamoji_state = st.text('Baiting numbers with candy...')
    # obj.transform()
    # alphamoji_state.text('Heres your AlphaMoji!')

    # st.text(obj.write_to_string())

    # obj.print_to_file("results/chester2_5_large.txt")




