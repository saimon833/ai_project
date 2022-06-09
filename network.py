import random
import numpy as np

class Network(object):

    def __init__(self, sizes):
        np.random.seed(572)
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.max_perf_inc = 1.04

    def __feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self.__sigmoid(np.dot(w, a)+b)
        return a

    def sse(self,_test_data):
        error=[pow(np.linalg.norm(self.__feedforward(x)-y),2) for (x,y) in _test_data]
        return 0.5*sum(error)    
    def mse(self,_test_data):
        error=[pow(np.linalg.norm(self.__feedforward(x)-y),2) for (x,y) in _test_data]
        return 1/len(_test_data)*sum(error)
 
    def train(self, training_data, test_data,epochs=1000, eta=0.1,
            error_goal=0.25,inc=1.05,dec=0.7,mini_batch_size=0):
        if mini_batch_size==0: mini_batch_size=len(training_data)
        n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                    for k in range(0, n, mini_batch_size)]
            old_error=self.sse(test_data)
            backup_weights = self.weights.copy()
            backup_biases = self.biases.copy()
            for mini_batch in mini_batches:
                self.__update_mini_batch(mini_batch, eta)
            new_error=self.sse(test_data)
            if new_error < error_goal:
                test=self.evaluate(test_data)
                test2=test/n_test*100
                print("Epoch {0}: {1:.2f}%".format(j+1, test2))
                return [j+1, test2]
            elif new_error < old_error:
                eta *= inc
            elif new_error > old_error * self.max_perf_inc:
                self.weights = backup_weights
                self.biases = backup_biases
                eta *= dec
            if j==epochs-1:
                test=self.evaluate(test_data)
                test2=test/n_test*100
                print("Epoch {0}: {1:.2f}%".format(j+1, test2))
                return [j+1, test2]
    def __update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.__backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/2)*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/2)*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def __backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.__sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.__cost_derivative(activations[-1], y) * self.__sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.__sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.__feedforward(x)), np.argmax(y))
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def __cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def __sigmoid(self,z):
        return 1.0/(1.0+np.exp(-z))

    def __sigmoid_prime(self,z):
        return self.__sigmoid(z)*(1-self.__sigmoid(z))
