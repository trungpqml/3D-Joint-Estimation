import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    )
)


def get_refine_net():
    from config import cfg
    model = RefineNet(cfg.joint_num)
    return model


if __name__ == "__main__":
    model = get_refine_net()
    model.compile(
        optimizer='sgd',
        loss='huber',
        metrics=['accuracy']
    )
    print(model.summary())
