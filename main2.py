import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# X, y = spiral_data(samples=100, classes=3)
#
# dense1 = Layer_Dense(2, 3)
#
# dense1.forward(X)
#
# print(dense1.output)


class Activtion_Softmax:
    def forward(self, inputs):
        max = np.max(inputs, axis=1, keepdims=True)
        print(max)
        subs = inputs - max
        print(subs)
        exp_values = np.exp(subs)
        print(exp_values)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


# inputs = np.array([[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])
# softmax = Activtion_Softmax()
# softmax.forward(inputs)
#
# print(softmax.output)

# inputs = np.array([4.8, 1.21, 2.385])
inputs = np.array([[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])
print(inputs)
max_inputs = np.max(inputs, axis=1, keepdims=False)
print(max_inputs)
print(inputs - max_inputs)
# ot = np.array([[8.9, -1.81, 0.2], [1.41, 1.051, 0.026]])
# print(inputs - ot)
