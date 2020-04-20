# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle

print(tf.__version__)

### data
with open('training_data.pickle', 'rb') as handle:
    (train_top_imgs, train_bottom_imgs, train_labels) = pickle.load(handle)

###  model
top_inputs = keras.layers.Input(shape=(50, 50, 3))
bottom_inputs = keras.layers.Input(shape=(70, 50, 3))

# top cnn
conv_1_T = keras.layers.Conv2D(32, (3, 3), activation='relu')(top_inputs)
maxpool_1_T = keras.layers.MaxPooling2D((2, 2))(conv_1_T)
conv_2_T = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_1_T)
maxpool_2_T = keras.layers.MaxPooling2D((2, 2))(conv_2_T)
conv_3_T = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_2_T)
flatten_T = keras.layers.Flatten()(conv_3_T)

# bottom cnn
conv_1_B = keras.layers.Conv2D(32, (3, 3), activation='relu')(bottom_inputs)
maxpool_1_B = keras.layers.MaxPooling2D((2, 2))(conv_1_B)
conv_2_B = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_1_B)
maxpool_2_B = keras.layers.MaxPooling2D((2, 2))(conv_2_B)
conv_3_B = keras.layers.Conv2D(64, (3, 3), activation='relu')(maxpool_2_B)
flatten_B = keras.layers.Flatten()(conv_3_B)

# shared network
flatten = keras.layers.concatenate([flatten_T, flatten_B])
dense_1 = keras.layers.Dense(64, activation='relu')(flatten)
outputs = keras.layers.Dense(2, activation='softmax')(dense_1)

# dense_1 = keras.layers.Dense(64, activation='relu')
# dense_2 = keras.layers.Dense(2)

# concate_1 = keras.layers.concatenate([y1,y2])
# y3 = dense_1(concate_1)
# outputs = dense_2(y3)

model = keras.Model(inputs=[top_inputs, bottom_inputs], outputs=outputs)
# model = keras.Model(inputs=[top_inputs], outputs=outputs)

model.compile(optimizer='adam',
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              loss=keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.fit([train_top_imgs, train_bottom_imgs], train_labels, epochs=10)

print("done")
