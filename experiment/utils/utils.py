import numpy as np


def logit(x):
    return np.log(x / (1 - x))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
