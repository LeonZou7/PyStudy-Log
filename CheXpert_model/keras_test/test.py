import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from model_test.keras_test import Load_Data

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

img_size = 320
channels = 3
num_classes = 3
batch_size = 16
num_filters = 32
kernel_size = 3

model = keras.Sequential()

model.add(keras.layers.Conv2D(filters=num_filters, 
                              kernel_size=kernel_size, 
                              strides=(2, 2), 
                              padding='SAME', 
                              activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(filters=num_filters, 
                              kernel_size=kernel_size, 
                              strides=(2, 2), 
                              padding='SAME', 
                              activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(keras.layers.Conv2D(filters=2*num_filters, 
                              kernel_size=kernel_size, 
                              strides=(2, 2), 
                              padding='SAME', 
                              activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Flatten())

num_output = 64

model.add(keras.layers.Dense(num_output, activation='relu'))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(2*num_output, activation='relu'))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datasets = Load_Data.init()


model.fit(datasets.train_dataset.)