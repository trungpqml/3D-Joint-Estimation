import numpy as np
from os.path import exists, join
from ...config import cfg

'''
TODO:
    1. Install Rope library for code refactor
    2. Check how to move data to GPU if exists
    3. Convert to Tensor, or Dataloader
'''


def load_train_data():
    trainset_in = np.load(cfg.training_in_path).reshape(-1, 54)
    trainset_gt = np.load(cfg.training_gt_path).reshape(-1, 54)
    return trainset_in, trainset_gt


def load_test_data():
    testset_in = np.load(cfg.testing_in_path).reshape(-1, 54)
    testset_gt = np.load(cfg.testing_gt_path).reshape(-1, 54)
    return testset_in, testset_gt
