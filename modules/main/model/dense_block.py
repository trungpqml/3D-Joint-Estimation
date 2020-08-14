from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Input
import tensorflow as tf
import pprint


class DenseBlock(Layer):
    '''
        A custom Dense block for RefineNet
    '''

    def __init__(self, units, alpha, p_dropout, activation=None, **kwargs):
        super().__init__(**kwargs)

        self.units = units,
        self.alpha = alpha,
        self.dropout_rate = p_dropout

        self.dense = Dense(
            units=units,
            name='DenseBlock_Dense',
            kernel_initializer='he_normal',
            bias_initializer='zeros',
        )
        self.dropout = Dropout(rate=p_dropout, name='DenseBlock_Dropout')
        self.batch_norm = BatchNormalization(name='DenseBlock_BN')

        if activation is None:
            self.activation = LeakyReLU(
                alpha=alpha, name='DenseBlock_Activation')
        else:
            self.activation = tf.keras.activations.get(activation)

    def call(self, inputs):
        h = self.dense(inputs)
        h = self.batch_norm(h)
        h = self.activation(h)
        output = self.dropout(h)
        return output

    # def build(self, input_shape):
    #     self.kernel = self.add_weight(
    #         name='kernel',
    #         shape=[input_shape[-1], self.units],
    #         initializer='he_normal'
    #     )
    #     self.bias = self.add_weight(
    #         name='bias',
    #         shape=[self.units],
    #         initializer='zeros'
    #     )
    #     super().build(input_shape)

    def build(self, input_shape):
        return super().build(input_shape)

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "units": self.units,
            "negative_slope": self.alpha,
            "dropout_rate": self.dropout_rate,
            "dense": tf.keras.layers.serialize(self.dense),
            "activation": tf.keras.activations.serialize(self.activation),
            "dropout": tf.keras.layers.serialize(self.dropout),
            "batch_norm": tf.keras.layers.serialize(self.batch_norm)
        }


if __name__ == "__main__":
    block = DenseBlock(2048, 0.01, 0.5)
    pprint.pprint(block.get_config())
