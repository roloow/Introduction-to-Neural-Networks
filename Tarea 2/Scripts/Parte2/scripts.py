#-------------------------------------------------------------------------------
# Name:        Scripts Parte 2
#
# Author:      Rolando Casanueva
#              Ricardo Carrasco
#
# Copyright:   (c) rCasanueva 2008-2017
#-------------------------------------------------------------------------------
# -*- coding: utf-8 -*-

"""
In order to run the code, uncomment the sections you want to see functioning
You can use ctrl + 1, to comment and uncomment some selected code.
"""
#######################################
#          Librerias
#######################################
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.models import load_model

#######################################
#        Lecturas Archivos
#######################################

train_data = sio.loadmat('train_32x32.mat')
test_data = sio.loadmat('test_32x32.mat')
X_train = train_data['X'].T
y_train = train_data['y'] - 1
X_test = test_data['X'].T
y_test = test_data['y'] - 1


#######################################
#       Determinar tamanno
#######################################

ntrain_data, n_channels, size_h, size_w = X_train.shape
ntest_data = X_test.shape[0]

#print "Imagenes de dimension:", size_h, "x", size_w

#######################################
#       Numero de clases
#######################################

n_classes = len(np.unique(y_train))

#print "Numero de clases:", n_classes

#######################################
#          Ver imagenes
#######################################

#IMG_NUM = 5

#TRAIN_EXAMPLES = random.sample(list(X_train), IMG_NUM)
#TEST_EXAMPLES = random.sample(list(X_test), IMG_NUM)

#for i in range(0, IMG_NUM):
#    sub = plt.subplot(2, IMG_NUM, i + 1)
#    image = TRAIN_EXAMPLES[i].transpose(2,1,0)
#    plt.imshow(image)
#    sub.get_xaxis().set_visible(False)
#    sub.get_yaxis().set_visible(False)

#plt.show()

#for i in range(0, IMG_NUM):
#    sub = plt.subplot(1, IMG_NUM, i + 1)
#    image = TEST_EXAMPLES[i].transpose(2,1,0)
#    plt.imshow(image)
#    sub.get_xaxis().set_visible(False)
#    sub.get_yaxis().set_visible(False)
#
#plt.show()

#######################################
#          Normalizacion0
#######################################

#X_train = X_train.astype('float16')
#X_test = X_test.astype('float16')
#
#X_train /= 225
#X_test /= 225
#
#Y_train = np_utils.to_categorical(y_train, n_classes)
#Y_test = np_utils.to_categorical(y_test, n_classes)

#######################################
#          Red Convolucional
#######################################

#model = Sequential()
#model.add(Convolution2D(16, (5, 5), border_mode='same', activation='relu', input_shape=(n_channels, size_h, size_w)))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Convolution2D(512, (7, 7), border_mode='same', activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(20, activation='relu'))
#model.add(Dense(n_classes, activation='softmax'))
#model.summary()

#adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
#model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['accuracy'])
#CNN = model.fit(X_train, Y_train, batch_size=1280, epochs=10, verbose=1, validation_data=(X_test, Y_test))

#plt.plot(range(1,11), CNN.history['loss'])

#CONV = [(5,5), (7,7), (9,9)]
#POOL = [(2,2), (4,4)]

"""
Tanto (D) como (E) se prueba manualmente los cambios, con el conjunto superior.
"""

#######################################
#           Cantidad de filtros
#######################################

#CAPA1 = [16, 32, 64]
#CAPA2 = [128, 256, 512]

#model = Sequential()
#model.add(Convolution2D(32, (3, 3), border_mode='same', activation='relu', input_shape=(n_channels, size_h, size_w)))
#model.add(Convolution2D(32, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Convolution2D(128, (3, 3), border_mode='same', activation='relu'))
#model.add(Convolution2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dense(20, activation='relu'))
#model.add(Dense(n_classes, activation='softmax'))
#model.summary()

#adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
#model.compile(loss='categorical_crossentropy', optimizer=adagrad, metrics=['accuracy'])
#CNN = model.fit(X_train, Y_train, batch_size=1280, epochs=10, verbose=1, validation_data=(X_test, Y_test))

"""
No se conocía este comando antes T-T  (.save())
"""
#model.save('Ideal.h5')

#######################################
#           Efecto de filtros
#######################################

#model = load_model('Ideal.h5')

#first = model.layers[0]
#weights = np.array(first.get_weights())

#print (weights[0].shape)
#print (weights)

#rand = random.randint(0, len(X_train) - 1)
#RANDOM_IMAGE = X_train[rand]
#plt.imshow(RANDOM_IMAGE[0].transpose(2, 1, 0))
#plt.show()


"""
Ver filtros con Keras backend (TODO)
"""


#######################################
#           Numeros Confusos
#######################################

#model = load_model('Ideal.h5')
#
#for i in range(10):
#    indexes = []
#    data = []
#    for idx, category in enumerate(y_test):
#        if category == i:
#            indexes.append(idx)
#    for idx in indexes:
#        data.append(X_test[idx])
#    data = np.array(data)
#    prediction = model.predict(data, verbose=1)
#    
#    numb, enum = np.unique(np.argmax(prediction, axis=1), return_counts=True)
#    dic = dict(zip(numb, enum))
#    sort = sorted(dic.values())
#    first = 0
#    second = 0
#    for k, v in dic.items():
#        if v == sort[-1]:
#            first = k
#        elif v == sort[-2]:
#            second = k
#    print ('Class', i, "gets: ", first, "and", second)
