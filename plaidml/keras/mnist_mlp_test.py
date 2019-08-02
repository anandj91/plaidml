# Original source: https://github.com/fchollet/keras/raw/master/examples/mnist_mlp.py
#
# This example has been slightly modified:
# ) It uses PlaidML as a backend (by invoking plaidml.keras.install_backend()).
# ) It removes Dropout.
# ) It's been reworked slightly to act as a functional integration test.
'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''
from __future__ import print_function

import sys

import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend

import numpy as np
np.random.seed(47)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


def load_data():
    integration_test_limit = 20000
    nb_classes = 10
    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_train = X_train[:integration_test_limit]
    y_train = y_train[:integration_test_limit]

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    return X_train, Y_train, X_test, Y_test


def build_model(use_batch_normalization=False, use_dropout=False):
    print('================= Sequential ====================')
    sys.stdout.flush()
    model = Sequential()
    print('================= Dense1 ====================')
    sys.stdout.flush()
    d = Dense(10, input_shape=(784,))
    model.add(d)
    print('================= Activation1 ====================')
    sys.stdout.flush()
    a = Activation('softmax')
    model.add(a)
    print('================= BUILD DONE ====================')
    sys.stdout.flush()

    return model


def run(use_batch_normalization=False, use_dropout=False):
    batch_size = 128
    epochs = 1

    # the data, shuffled and split between train and test sets
    X_train, Y_train, X_test, Y_test = load_data()
    X_train = X_train[:batch_size,:]
    Y_train = Y_train[:batch_size,:]
    print(X_train.shape, 'train samples')
    print(X_test.shape, 'test samples')

    print('================= BUILD ====================')
    sys.stdout.flush()
    model = build_model(use_batch_normalization=use_batch_normalization, use_dropout=use_dropout)

    print('================= SUMMARY ====================')
    sys.stdout.flush()
    model.summary()

    print('================= COMPILE ====================')
    sys.stdout.flush()
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    print('================= FIT ====================')
    sys.stdout.flush()
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    print('================= DONE ====================')
    sys.stdout.flush()


if __name__ == '__main__':
    plaidml._internal_set_vlog(4)
    run()
