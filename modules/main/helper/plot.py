import matplotlib.pyplot as plt
from ...config import cfg
from os.path import join, exists
from os import makedirs


def plot_fig(i, history):
    fig = plt.figure()
    plt.plot(range(1, epochs+1),
             history.history['val_acc'], label='validation')
    plt.plot(range(1, epochs+1), history.history['acc'], label='training')
    plt.legend(loc=0)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.xlim([1, epochs])
#     plt.ylim([0,1])
    plt.grid(True)
    plt.title("Model Accuracy")
    plt.show()

    image_path = join(cfg.vis_dir
    fig.savefig('img/'+str(i)+'-accuracy.jpg')
    plt.close(fig)
