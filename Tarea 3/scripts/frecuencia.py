import keras
import numpy as np

from keras.datasets import imdb
from matplotlib import pyplot as plt

np.random.seed(3)
(X_train, y_train), (X_test, y_test) = imdb.load_data(path="B:\Programas\Google Drive\USM\Redes Neuronales\Tarea 3\datasets\imdb.npz",seed=15)

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)


# unique, counts = np.unique(np.hstack(X), return_counts=True)

# plt.plot(unique, counts)
# plt.ylabel("Frequency")
# plt.xlabel("Index of word")
# plt.show()
#
#
# lunique = map(np.log, unique)
# lcounts = map(np.log, counts)
# plt.plot(lunique, lcounts)
# plt.ylabel("Log(F)")
# plt.xlabel("Log(IoW)")
# plt.show()

good = []
bad = []
for i in range(len(y)):
    if y[i] == 1:
        good.append(X[i])
    else:
        bad.append(X[i])

if (len(good) + len(bad)) == len(X):
    good = np.asarray(good)
    bad = np.asarray(bad)

    uniqueG, countsG = np.unique(np.hstack(good), return_counts=True)
    uniqueB, countsB = np.unique(np.hstack(bad), return_counts=True)

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.plot(uniqueG, countsG)
    ax1.set_title("Positivos")
    ax2.plot(uniqueB, countsB)
    ax2.set_title("Negativos")
    # ax1.ylabel("Frequency")
    # ax1.xlabel("Index of word")
    # ax2.ylabel("Frequency")
    # ax2.xlabel("Index of word")
    plt.show()


    luniqueG = map(np.log, uniqueG)
    lcountsG = map(np.log, countsG)
    luniqueB = map(np.log, uniqueB)
    lcountsB = map(np.log, countsB)
    f, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.plot(luniqueG, lcountsG)
    ax1.set_title("Positivos")
    # ax1.ylabel("Log(F)")
    # ax1.xlabel("Log(IoW)")
    ax2.plot(luniqueB, lcountsB)
    ax2.set_title("Negativos")
    # ax2.ylabel("Log(F)")
    # ax2.xlabel("Log(IoW)")
    plt.show()
