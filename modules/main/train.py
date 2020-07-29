from .data.load_data import load_data

from ..config import cfg
from .helper.loss import BoneLoss
from .helper.optimizer import get_optimizier
from .helper.transformation import get_transformation
from .model.model import get_refine_net

import matplotlib.pyplot as plt
import numpy as np
import time


def train():
    pass


if __name__ == "__main__":
    train()
