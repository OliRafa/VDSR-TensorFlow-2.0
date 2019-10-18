import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Conv2D, Add


class VDSR(Model):
    def __init__(self, layers=20):
        super(VDSR, self).__init__()
        self.input = Conv2D(
            input_shape=(128, 128, 3),
            filters=64,
            kernel=(3, 3),
            stride=(1, 1),
            activation='relu',
            padding='same'
        )
        self.conv_layers = self._generate_layers(layers)
        self.output = Conv2D(1, (3, 3), 1)

    def _generate_layers(self, layers):
        block = Sequential()
        for i in range(layers):
            block.add(Conv2D(64, (3, 3), 1, activation='relu', padding='same'))
        return block

    def call(self, inputs):
        residual = inputs
        output = self.input(inputs)
        output = self.conv_layers(output)
        output = self.output(output)
        return Add([output, residual])