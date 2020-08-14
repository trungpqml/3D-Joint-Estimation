import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Sequential
from dense_block import DenseBlock
import pprint


class ResidualBlock(Layer):
    def __init__(self, n_stages, units, alpha, p_dropout, **kwargs):
        super().__init__(**kwargs)

        self.num_stages = n_stages

        self.hidden = Sequential()

        for _ in range(n_stages):
            self.hidden.add(DenseBlock(units, alpha, p_dropout))

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "num_stages": self.num_stages,
            "hidden": self.hidden.get_config()
        }


if __name__ == "__main__":
    res_block = ResidualBlock(1, 2048, 0.01, 0.5)
    pprint.pprint(res_block.get_config())
