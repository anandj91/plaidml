import numpy as np

import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend as K

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Layer

base_type = plaidml.DType.FLOAT32
new_type = plaidml.DType.CUSTOM

#plaidml._internal_set_vlog(4)

# the four different states of the XOR gate
training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

# the four expected results in the same order
target_data = np.array([[0],[1],[1],[0]], "float32")

class Cast(Layer):
    def __init__(self, **kwargs):
        super(Cast, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Cast, self).build(input_shape)

    def call(self, x): 
        return K.cast(x, self.dtype)

model = Sequential()
model.add(Cast(input_shape=(2,), dtype=new_type))
model.add(Dense(16, activation='relu', dtype=new_type))
model.add(Dense(1, activation='sigmoid', dtype=new_type))
model.add(Cast(dtype=base_type))

model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['binary_accuracy'])

model.fit(training_data, target_data, epochs=100, verbose=2)

print('PREDICTION', model.predict(training_data))
