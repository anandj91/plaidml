'''Trains a simple deep NN on the MNIST dataset.

Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import plaidml
import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend as K
#plaidml._internal_set_vlog(4)

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Layer
from keras.optimizers import RMSprop
from keras.utils import np_utils

batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)

base_type = plaidml.DType.FLOAT32
new_type = plaidml.DType.FLOAT32

class Cast(Layer):
    def __init__(self, **kwargs):
        super(Cast, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Cast, self).build(input_shape)

    def call(self, x): 
        return K.cast(x, self.dtype)

model = Sequential()
#model.add(Cast(input_shape=(784,), dtype=new_type))
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
#model.add(Cast(dtype=base_type))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
