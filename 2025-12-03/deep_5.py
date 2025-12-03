# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 13:51:09 2025

@author: Choe JongHyeon
"""

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

x = np.arange(0, 1000)
y = np.arange(0, 1000)

model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=32, verbose=1)


model = keras.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=(1,)),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x, y, epochs=100, batch_size=32, verbose=1)