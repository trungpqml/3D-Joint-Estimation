import sys
import os
import pprint
from .refine_net import RefineNet

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.abspath(__file__), '..', '..', '..')))


def get_refine_net():
    from config import cfg
    model = RefineNet(cfg.joint_num)
    return model


if __name__ == "__main__":
    model = get_refine_net()
    # pprint.pprint(model.summary())
    print("Get model module!")
