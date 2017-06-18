#-------------------------------------------------------------------------------
# Name:        Scripts Parte 2
#
# Author:      Rolando Casanueva
#              Ricardo Carrasco
#
# Copyright:   (c) rCasanueva 2008-2017
#-------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random
import math
import pickle
import os
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, rmsprop
from keras.models import load_model

def load_CIFAR_one(filename):
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(PATH):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(PATH, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_one(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_one(os.path.join(PATH, 'test_batch'))
    return Xtr, Ytr, Xte, Yte

x_train, y_train, x_test, y_test = load_CIFAR10('')

class_names = ['Avion', 'Auto', 'Ave', 'Gato', 'Venado', 'Perro', 'Rana', 'Caballo', 'Barco', 'Camion']

x_train = x_train.reshape((x_train.shape[0],32,32,3), order="F")
x_test= x_test.reshape((x_test.shape[0],32,32,3), order="F")

num_classes = 10

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Convolution2D(64, (3, 3), padding='same', input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()
#
#model.save('cnn1.h5')

def step_decay(epoch):
    initial_lrate = 0.001
    lrate = initial_lrate * math.pow(0.5, math.floor((1+epoch)/5))
    lrate = max(lrate,0.00001)
    return lrate

#model = load_model('cnn1.h5')
#
#batch_size = 100
#opt = SGD(lr=0.0, momentum=0.9, decay=0.0)
#lrate = LearningRateScheduler(step_decay)
#
#model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#model.fit(x_train, y_train, batch_size=batch_size, epochs=25, validation_data=(x_test, y_test), shuffle=True,callbacks=[lrate], verbose= True)
#
#model.save('cnn_entrenada.h5')



model = load_model('cnn1.h5')
opt = rmsprop(lr=0.001, decay=1e-6)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
CNN = model.fit(x_train, y_train,batch_size=100,nb_epoch=25, validation_data=(x_test, y_test),shuffle=True)

model.save('cnn_entrenada2.h5')