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
import plaidml.keras.backend as K

import numpy as np
np.random.seed(47)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Layer
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

#plaidml._internal_set_vlog(4)

base_type = plaidml.DType.FLOAT32
new_type = plaidml.DType.CUSTOM

class Cast(Layer):
    def __init__(self, **kwargs):
        super(Cast, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Cast, self).build(input_shape)

    def call(self, x):
        return K.cast(x, self.dtype)


def load_data():
    integration_test_limit = 128
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
    model = Sequential()
    model.add(Cast(input_shape=(784,), dtype=new_type))
    model.add(Dense(128, input_shape=(784,), dtype=new_type))
    model.add(Activation('relu', dtype=new_type))
    if use_batch_normalization:
        model.add(BatchNormalization(dtype=new_type))
    if use_dropout:
        model.add(Dropout(0.2, dtype=new_type))
    model.add(Dense(128, dtype=new_type))
    model.add(Activation('relu', dtype=new_type))
    if use_batch_normalization:
        model.add(BatchNormalization(dtype=new_type))
    if use_dropout:
        model.add(Dropout(0.2, dtype=new_type))
    model.add(Dense(10, dtype=new_type))
    model.add(Activation('softmax', dtype=new_type))
    model.add(Cast(dtype=base_type))

    return model


def run(use_batch_normalization=False, use_dropout=False):
    batch_size = 128
    epochs = 1

    # the data, shuffled and split between train and test sets
    X_train, Y_train, X_test, Y_test = load_data()
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    model = build_model(use_batch_normalization=use_batch_normalization, use_dropout=use_dropout)

    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    #score = model.evaluate(X_test, Y_test, verbose=1)
    score = (0, 0)

    return score


if __name__ == '__main__':
    score = run()

    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    if .75 < score[1]:
        sys.exit(0)
    sys.exit(1)
