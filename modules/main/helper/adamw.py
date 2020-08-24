import tensorflow as tf
import math
from tensorflow.keras.optimizers import Adam
import pprint


class AdamW(Adam):
    def __init__(
        self,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-08,
        decay=0.0,
        amsgrad=False,
        name='AdamW',
        **kwargs
    ):
        if not 0.0 <= learning_rate:
            raise ValueError(f'Invalid learning rate:{learning_rate}')
        if not 0.0 <= epsilon:
            raise ValueError(f'Invalid epsilon value: {epsilon}')
        super().__init__(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
            name=name,
            decay=decay,
            **kwargs
        )

    def minimize(self, loss, var_list, grad_loss=None, name=None):
        return super().minimize(loss, var_list, grad_loss=grad_loss, name=name)

    # def __setstate__(self, state):
    #     super(AdamW, self).__setstate__(state)
    #     for group in self.param_groups:
    #         group.setdefault('amsgrad', False)


if __name__ == "__main__":
    optimizer = AdamW()
    pprint.pprint(optimizer.get_config())
