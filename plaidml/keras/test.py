import sys

import plaidml.keras
plaidml.keras.install_backend()
import plaidml.keras.backend as K

from keras.models import Sequential
from keras.layers import Layer, Dense, Activation
import numpy as np

class Cast(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Cast, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Cast, self).build(input_shape)

    def call(self, x):
        return K.cast(x, self.dtype)

class Dota(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Dota, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='zeros',
                                      #initializer='glorot_uniform',
                                      trainable=True)
        super(Dota, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        r = K.dot(x, self.kernel)
        return r

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

#plaidml._internal_set_vlog(4)
#plaidml.set_backtrace(True)

# Generate dummy data
x_train = np.random.random((128, 20))
y_train = np.random.randint(2, size=(128, 1))
x_test = np.random.random((100, 20))
y_test = np.random.randint(2, size=(100, 1))

new_type = plaidml.DType.CUSTOM
#new_type = plaidml.DType.FLOAT16
base_type = plaidml.DType.FLOAT32

model = Sequential()
model.add(Cast(20, dtype=new_type))
#model.add(Dota(1, dtype=new_type))
model.add(Dense(1, input_dim=20, activation='sigmoid', dtype=new_type))
model.add(Cast(20, dtype=base_type))

model.compile(optimizer='sgd', loss='mse')

print('MODEL', model.to_json())
sys.stdout.flush()

#model.predict(x_train, batch_size=128)
model.fit(x_train, y_train, epochs=20, batch_size=128)
