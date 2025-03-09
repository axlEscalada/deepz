import numpy as np

# import nnfs
# from nnfs.datasets import spiral_data
#
# nnfs.init()
#


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, fixed_weights=None, fixed_biases=None):
        if fixed_weights is not None:
            self.weights = fixed_weights
        else:
            self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)

        if fixed_biases is not None:
            self.biases = fixed_biases
        else:
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


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X = [
    [0.00000000, 0.00000000],
    [0.00073415, 0.01007430],
    [0.00431511, 0.01973579],
    [0.02011100, 0.02266763],
]


fixed_weights1 = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
fixed_biases1 = np.array([[0.1, 0.2, 0.3]])

fixed_weights2 = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]])
fixed_biases2 = np.array([[0.1, 0.2, 0.3]])

dense1 = Layer_Dense(2, 3, fixed_weights1, fixed_biases1)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3, fixed_weights2, fixed_biases2)

activation2 = Activation_Softmax()
# dense1 = Layer_Dense(2, 3)
# activation1 = Activation_ReLU()
# dense2 = Layer_Dense(3, 3)
# activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output)
