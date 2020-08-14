import os
from os import makedirs
from os.path import join, exists, dirname, abspath


class Config(object):
    # dataset
    training_in_filename = 'data_train_in.npy'
    training_gt_filename = 'data_train_gt.npy'
    testing_in_filename = 'data_test_int.npy'
    testing_gt_filename = 'data_test_gt.npy'

    # directory
    current_dir = dirname(abspath(__file__))
    root_dir = join(current_dir, '..')
    data_dir = join(root_dir, 'data')
    output_dir = join(current_dir, 'output')
    vis_dir = join(output_dir, 'vis')
    log_dir = join(output_dir, 'log')
    best_model_dir = join(output_dir, 'best_model')

    # input, output
    training_in_path = join(data_dir, training_in_filename)
    training_gt_path = join(data_dir, training_gt_filename)
    testing_in_path = join(data_dir, testing_in_filename)
    testing_gt_path = join(data_dir, testing_gt_filename)

    best_model_path = join(best_model_dir, 'best_model.h5')
    joint_num = 18

    # training second model config
    epochs = 250
    train_batch_size = 256
    learning_rate = 1e-3
    learning_rate_decay = 0.96
    learning_rate_decay_step = 50
    optimizer = 'adamW'
    weight_decay = 1e-5
    bone_lambda = 10.
    joint_lambda = 1.
    scheduler = 'cyclicLR'
    seed = 0
    patience = 20

    # testing config
    test_batch_size = 64

    # others
    num_thread = 16
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False
    first_model = 'resnext'
    best_model = 0
    test_subsampling_ratio = 5

    def set_args(self, gpu=gpu_ids, continue_train=False, first_model=first_model, best_model=best_model, ratio=test_subsampling_ratio, test_batch_size=test_batch_size):
        self.gpu_ids = gpu
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        self.first_model = first_model
        self.best_model = best_model
        self.test_subsampling_ratio = ratio
        self.test_batch_size = test_batch_size
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))


cfg = Config()

if __name__ == "__main__":
    print(cfg)
