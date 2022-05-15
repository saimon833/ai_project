import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        pass
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
