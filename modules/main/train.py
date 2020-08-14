# All configuration
from ..config import cfg

# Data loader
from .data.load_data import load_train_data

# Helper function
from .helper.callback import get_callbacks
from .helper.loss import get_loss_function
from .helper.optimizer import get_optimizer, get_scheduler
from .helper.plot import plot_history
from .helper.transformation import get_transformation

# Model
from .model.model import get_refine_net

# Other
import matplotlib.pyplot as plt
import numpy as np
import time


def train():
    # Check how to split train data for validation
    trainset_in, trainset_gt = load_train_data()
    model = get_refine_net(cfg, True)
    loss = get_loss_function()
    optimizer = get_optimizier(optimizer_name)
    scheduler = get_scheduler(scheduler_name)
    calback_list = get_callbacks()

    model.compile(
        optimizer=optimizer,
        loss=joint_loss,
        metrics=['accuracy']
    )

    history = model.fit(
        trainset_in,
        trainset_gt,
        epochs=cfg.epochs,
        callbacks=calback_list,
        validation_split=0.2,
        verbose=1
    )

    plot_history(history)


if __name__ == "__main__":
    train()
