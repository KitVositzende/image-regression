#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 19:52:12 2020

@author: shijiayang
"""

from skimage.io import imread, imsave
import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf


filename = "landscape.png"
image = imread(filename, as_grey=False, plugin='pil')
#image = np.divide(image, 255.0)
image = image/255.0
plt.subplot(311)
plt.imshow(image)
img_height, img_width = image.shape[0],image.shape[1]
num_channels = image.shape[2]

x = []
y = []
for i in range(img_height):
    for j in range(img_width):
        x.append([i / img_height, j / img_width])
        y.append(image[i][j])
x = np.array(x)
y = np.array(y)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')<0.05):
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()
model = tf.keras.models.Sequential([
    #tf.keras.layers.Flatten()
    tf.keras.layers.Dense(30,input_dim=2,activation='relu'),
    tf.keras.layers.Dense(30,activation='relu'),
    tf.keras.layers.Dense(30,activation='relu'),
    tf.keras.layers.Dense(30,activation='relu'),
    tf.keras.layers.Dense(num_channels,activation='relu'),
    ])

model.compile(
    loss='mean_squared_error',
    optimizer='rmsprop',
     metrics=['accuracy']
)

history = model.fit(
    x,
    y,
    batch_size=128,
    epochs=5000,
    shuffle=True,
    callbacks=[callbacks]
)

predicted_image = model.predict(x)
predicted_image = np.clip(predicted_image, 0, 1)
predicted_image = predicted_image.reshape(image.shape)

plt.subplot(312)
plt.imshow(predicted_image)

plt.subplot(313)
loss = history.history['loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label='Training Loss')
