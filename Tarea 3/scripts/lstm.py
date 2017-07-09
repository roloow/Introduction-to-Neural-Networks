import keras
import numpy as np

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

np.random.seed(3)

# (X_train, y_train), (X_test, y_test) = imdb.load_data(path="B:/Programas/Google Drive/USM/Redes Neuronales/Tarea 3/datasets/imdb.npz", num_words=3000, seed=15)
#
# X_train = sequence.pad_sequences(X_train, maxlen=500)
# X_test = sequence.pad_sequences(X_test, maxlen=500)
#
#
# TOP_WORDS = 3000
# EVL = 64
# MAX_WORDS = 500
#
# model = Sequential()
# model.add(Embedding(TOP_WORDS, EVL, input_length=MAX_WORDS))
# model.add(LSTM(100))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
#
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: {0}".format(scores[1]*100))

# model.save('lstm.h5')
# model.save('lstmv2.h5')
#
# for i in range(1,11):
#     (X_train, y_train), (X_test, y_test) = imdb.load_data(path="B:/Programas/Google Drive/USM/Redes Neuronales/Tarea 3/datasets/imdb.npz", num_words=1000*i, seed=15)
#
#     X_train = sequence.pad_sequences(X_train, maxlen=500)
#     X_test = sequence.pad_sequences(X_test, maxlen=500)
#
#
#     TOP_WORDS = 1000*i
#     EVL = 64
#     MAX_WORDS = 500
#
#     model = Sequential()
#     model.add(Embedding(TOP_WORDS, EVL, input_length=MAX_WORDS))
#     model.add(LSTM(100))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     print(model.summary())
#
#     model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
#     model.save('lstm_tw_'+str(i)+'.h5')


(X_train, y_train), (X_test, y_test) = imdb.load_data(path="B:/Programas/Google Drive/USM/Redes Neuronales/Tarea 3/datasets/imdb.npz", num_words=3000, seed=15)

X_train = sequence.pad_sequences(X_train, maxlen=500)
X_test = sequence.pad_sequences(X_test, maxlen=500)


TOP_WORDS = 3000
EVL = 32
MAX_WORDS = 500

model = Sequential()
model.add(Embedding(TOP_WORDS, EVL, input_length=MAX_WORDS))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
