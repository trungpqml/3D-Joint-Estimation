import tensorflow as tf
from tensorflow.keras.Model import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.losses import Huber, MAE, MSE
from tensorflow.keras.callbacks import Callback
from ..config import cfg


class Linear(tf.keras.Model):
    def __init__(self, linear_size, alpha=0.01, p_dropout=0.5, **kwargs):
        super().__init__(**kwargs)
        self.linear_size = linear_size
        self.p_dropout = p_dropout

        self.relu = LeakyReLU(alpha=alpha)
        self.dropout = Dropout(rate=self.p_dropout)

        self.dense_1 = Dense(self.linear_size, self.linear_size)
        self.batch_norm_1 = BatchNormalization()

        self.dense_2 = Dense(self.linear_size, self.linear_size)
        self.batch_norm_2 = BatchNormalization()

    def call(self, input):
        y = self.dense_1(input)
        y = self.batch_norm_1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.dense_2(input)
        y = self.batch_norm_2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y
        return out


class RefineNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        pass


def get_refine_net(cfg, is_train):
    model = RefineNet()
    return model
