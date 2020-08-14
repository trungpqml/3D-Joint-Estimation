from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU, Input
from residual_block import ResidualBlock
import pprint


class RefineNet(Model):
    def __init__(self, joint_num, linear_size=2048, num_stage=3, p_dropout=0.5, alpha=0.01, **kwargs):
        super().__init__(**kwargs)

        self.input_size, self.output_size = joint_num * 3, joint_num * 3

        self.dense_in = Dense(
            units=linear_size,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            input_shape=(self.input_size,),
            name='inputs_layer')
        self.activation = LeakyReLU(alpha=alpha)
        self.dropout = Dropout(p_dropout)

        self.linear_stages = ResidualBlock(
            num_stage, linear_size, alpha, p_dropout)

        self.dense_out = Dense(
            units=self.output_size,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            name='outputs_layer'
        )

    def call(self, inputs):
        h = self.dense_in(inputs)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.linear_stages(h)
        outputs = self.dense_out(h)


if __name__ == "__main__":
    model = RefineNet(18)
    print(model.summary())
