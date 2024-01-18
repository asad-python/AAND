import random
import pickle
import numpy as np
from os import path
import logging
from PIL import Image
from keras.datasets import fashion_mnist
from defaults import get_cfg_defaults

def get_fashion_mnist():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    images = np.concatenate((train_images, test_images))
    labels = np.concatenate((train_labels, test_labels))

    assert(images.shape == (70000, 28, 28))

    _images = []
    for im in images:
        im = Image.fromarray(im)
        im = im.resize((32, 32), resample=Image.BILINEAR)
        im = np.array(im)
        _images.append(im)
    images = np.asarray(_images)

    assert(images.shape == (70000, 32, 32))

    return [(l, im) for l, im in zip(labels, images)]


def partition(cfg, logger):
    random.seed(0)
    fashion_mnist = get_fashion_mnist()

    random.shuffle(fashion_mnist)

    folds = cfg.DATASET.FOLDS_COUNT

    class_bins = {}
    for x in fashion_mnist:
        if x[0] not in class_bins:
            class_bins[x[0]] = []
        class_bins[x[0]].append(x)

    fashion_mnist_folds = [[] for _ in range(folds)]
    for _class, data in class_bins.items():
        count = len(data)
        logger.info("Class %d count: %d" % (_class, count))
        count_per_fold = count // folds
        for i in range(folds):
            fashion_mnist_folds[i] += data[i * count_per_fold: (i + 1) * count_per_fold]

    logger.info("Folds sizes:")
    for i in range(len(fashion_mnist_folds)):
        print(len(fashion_mnist_folds[i]))

        output = open(path.join(cfg.DATASET.PATH, 'data_fold_%d.pkl' % i), 'wb')
        pickle.dump(fashion_mnist_folds[i], output)
        output.close()


if __name__ == "__main__":
    cfg = get_cfg_defaults()
    cfg.merge_from_file('configs/fmnist.yaml')  # Point to Fashion MNIST Config
    cfg.freeze()
    logger = logging.getLogger("logger")
    partition(cfg, logger)
