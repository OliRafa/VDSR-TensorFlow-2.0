import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, Add


class Vdsr(Model):
    def __init__(self, layers=20, weight_decay=0.0001):
        super(Vdsr, self).__init__()
        self.input_layer = Conv2D(
            input_shape=(41, 41, 3),
            filters=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            activation='relu',
            padding='same',
            kernel_regularizer=keras.regularizers.l2(weight_decay)
        )
        self.conv_layers = self._generate_layers(layers, weight_decay)
        self.output_layer = Conv2D(1, (3, 3), 1, padding='same')

    def _generate_layers(self, layers, weight_decay):
        block = Sequential()
        for i in range(layers):
            block.add(Conv2D(
                filters=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                activation='relu',
                padding='same',
                kernel_regularizer=keras.regularizers.l2(weight_decay)
                ))
        return block

    def call(self, inputs):
        residual = inputs
        output = self.input_layer(inputs)
        output = self.conv_layers(output)
        output = self.output_layer(output)
        return Add()([output, residual])
