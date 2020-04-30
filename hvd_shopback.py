from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.losses import sparse_categorical_crossentropy
import time
# from tensorflow.keras.models import load_model, Sequential,Model 
# from tensorflow.keras.layers import Input, Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.losses import sparse_categorical_crossentropy

import math
import tensorflow as tf
import horovod.keras as hvd

# Horovod: initialize Horovod.
hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(hvd.local_rank())
K.set_session(tf.Session(config=config))

batch_size = 128
num_classes = 2

# Horovod: adjust number of epochs based on number of GPUs.
epochs = 50
epochs = int(math.ceil(epochs / hvd.size()))


# The data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

import pickle
with open("xiaoming_dataset_xy_120_full.pickle", 'rb') as handle:
    (x,y) = pickle.load(handle)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Input image dimensions
img_rows, img_cols = x_train.shape[1], x_train.shape[2]
channels = 3

# print(y_train.shape)
# exit()

#if K.image_data_format() == 'channels_first':
#    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
#    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
#    input_shape = (channels, img_rows, img_cols)
#else:
#    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
#    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
#    input_shape = (img_rows, img_cols, channels)

input_shape = (img_rows, img_cols, channels)
print("inputshape", input_shape)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')


# Convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape, "train label")

# exit()
def get_model(input_shape=(120,50,3)):
    model2 = Sequential()
    model2.add(Conv2D(16,(2,2),strides=(1,1), padding = 'same', input_shape=input_shape))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    model2.add(Conv2D(64,(2,2),strides=(1,1), padding = 'same'))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2, 2), padding = 'same'))
    model2.add(Conv2D(256,(2,2),strides=(1,1), padding = 'same'))
    model2.add(Activation('relu'))
    model2.add(Flatten())
    model2.add(Dense(256,activation='relu', kernel_initializer='he_normal'))
    model2.add(Dropout(0.3))
    model2.add(Dense(64,activation='relu', kernel_initializer='he_normal'))
    model2.add(Dropout(0.3))
    model2.add(Dense(2,kernel_initializer='he_normal'))
    model2.add(Activation('softmax'))
    return model2

model = get_model(input_shape=input_shape)

# Horovod: adjust learning rate based on number of GPUs.
# opt = keras.optimizers.Adadelta(1.0 * hvd.size())
opt=keras.optimizers.Adam(lr=0.0001* hvd.size(), beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Horovod: add Horovod Distributed Optimizer.
opt = hvd.DistributedOptimizer(opt)

model.compile(optimizer=opt,
              #loss=keras.losses.categorical_crossentropy,
              loss=sparse_categorical_crossentropy,
              metrics=['accuracy'])

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
]

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

start=time.time()

# x_train=x_train[hvd.rank()::hvd.size()]
# y_train=y_train[hvd.rank()::hvd.size()]

model.fit(x_train, y_train,
          batch_size=batch_size,
          callbacks=callbacks,
          epochs=epochs,
          verbose=1 if hvd.rank() == 0 else 0,
          validation_data=(x_test, y_test))
print("training used %d sec" % (time.time()-start))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print(model.summary())
print(x_train.shape[1])
