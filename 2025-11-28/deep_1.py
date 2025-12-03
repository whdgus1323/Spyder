# -*- coding: utf-8 -*-
"""
Created on Wed Dec  3 10:46:35 2025

@author: Choe JongHyeon
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except:
        pass

X = np.random.rand(1000, 28, 28, 1).astype(np.float32)
y = np.random.randint(0, 10, 1000)

model = keras.Sequential([
    keras.layers.Conv2D(16, 3, activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, 3, activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

pred = model.predict(X[:9]).argmax(axis=1)

plt.figure(figsize=(6,6))
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X[i].reshape(28,28), cmap='gray')
    plt.title(f"pred: {pred[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
