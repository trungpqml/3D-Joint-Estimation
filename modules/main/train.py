from .data.load_data import load_train_data

from ..config import cfg
from .helper.callbacks import get_callbacks
from .helper.loss import get_loss_function
from .helper.optimizer import get_optimizier, get_scheduler
from .helper.plot import plot_history
from .helper.transformation import get_transformation
from .model.model import get_refine_net

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
        validation_data=(trainset_in_val, trainset_gt_val),
        verbose=1
    )

    plot_history(history)


if __name__ == "__main__":
    train()
