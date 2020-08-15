'''
TODO:
    1. Print validation loss after each epoch
    2. Print learning rate after each epoch
    3. Print model hyper parameter after each epoch
    4. Save best model after improvement using: model.save('<model_name.h5>')
    5. TensorBoard
    6. EarlyStopping callback
'''
import sys
import os
from os.path import join, exists
from os import makedirs
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, Callback

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.realpath(__file__)
            )
        )
    )
)


def get_run_logdir():
    import time
    from config import cfg
    run_id = time.strftime('run_%Y_%m_%d-%H_%M_%S')
    run_logdir = join(cfg.log_dir, run_id)
    return run_logdir


def get_tensorboard_callback():
    run_logdir = get_run_logdir()
    return TensorBoard(run_logdir)


def get_checkpoint_callback():
    from config import cfg
    if not exists(cfg.best_model_path):
        makedirs(cfg.best_model_path)

    return ModelCheckpoint(
        filepath=cfg.best_model_path,
        save_best_only=True
    )


def get_earlystopping_callback():
    from config import cfg
    return EarlyStopping(
        patience=cfg.patience,
        restore_best_weights=True
    )


class LogCallback(Callback):
    def on_epoch_begin(self, epoch, logs):
        '''Print learning rate'''
        print()

    def on_epoch_end(self, epoch, logs):
        '''Print current training accuracy'''
        print()

    def on_train_begin(self):
        '''Print model summary'''
        print()

    def on_train_end(self):
        '''Plot training loss, validation loss, training accuracy, testing accuracy'''
        print()


def get_logging_callback():
    return LogCallback()


def get_callbacks():
    callback_list = []
    callback_list.append(get_tensorboard_callback())
    callback_list.append(get_checkpoint_callback())
    callback_list.append(get_earlystopping_callback())
    callback_list.append(get_logging_callback())
    return callback_list


if __name__ == "__main__":
    callback_list = get_callbacks()
    print(*callback_list)
