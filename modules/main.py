import tensorflow as tf
import numpy as np
import time
from train import train
from test import test

if __name__ == "__main__":
    print("Start training model")
    train()
    test()
