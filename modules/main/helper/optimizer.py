from tensorflow.keras.optimizers import SGD, Adam, Nadam, schedules
from tensorflow.keras.callbacks import LearningRateScheduler
from ...config import cfg
import math


def step_decay(epoch):
    initial_learning_rate = cfg.learning_rate
    drop = cfg.learning_rate_decay
    epochs_drop = cfg.learning_rate_decay_step
    learning_rate = initial_learning_rate * \
        math.pow(drop, math.floor((epoch)/epochs_drop))
    return learning_rate


def get_optimizier(optimizer_name, scheduler_name, model):
    # Optimizer
    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=cfg.learning_rate)
    elif optimizer_name == 'nadam':
        optimizer = Nadam(learning_rate=cfg.learning_rate)
    elif optimizer == '':
        optimizer = SGD(
            learning_rate=cfg.learning_rate,
            momentum=0.9,
            nesterov=True
        )
    else:
        print(f"Error! Unknown optimizer name: {optimizer_name}")
        assert 0

    # Scheduler
    if scheduler_name == 'stepLR':
        scheduler = LearningRateScheduler(step_decay)
    elif scheduler_name == 'cyclicLR':
        # scheduler = schedules.ExponentialDecay()
        pass
    else:
        print("Error! Unkown scheduler name: ", scheduler_name)
        assert 0

    return optimizer, scheduler
