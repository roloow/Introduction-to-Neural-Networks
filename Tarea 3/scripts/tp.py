import keras
import numpy as np

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import load_model
from matplotlib import pyplot as plt

np.random.seed(3)

X = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
Y = []

for i in range(1,9):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(path="B:/Programas/Google Drive/USM/Redes Neuronales/Tarea 3/datasets/imdb.npz", num_words=1000*i, seed=15)

    X_train = sequence.pad_sequences(X_train, maxlen=500)
    X_test = sequence.pad_sequences(X_test, maxlen=500)

    model = load_model('B:/Programas/Google Drive/USM/Redes Neuronales/Tarea 3/scripts/lstm_tw_'+str(i)+'.h5')
    scores = model.evaluate(X_test, y_test, verbose=0)
    Y.append(scores[1]*100)



plt.plot(X,Y)
plt.xlabel("Top Words Quantity")
plt.ylabel("Accuracy")
plt.show()
