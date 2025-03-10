import numpy as np

import nnfs
from nnfs.datasets import spiral_data

nnfs.init()


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


class Loss:
    def calculate(self, output, y):
        sample_loss = self.forward(output, y)
        print("sample_loss:", sample_loss)
        data_loss = np.mean(sample_loss)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        print("samples:", samples)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        print("y_clipped:", y_pred_clipped)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
            print("correc_confidences:", correct_confidences)
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
            print("correc_confidences:", correct_confidences)

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# X = [
#     [0.00000000, 0.00000000],
#     [0.00073415, 0.01007430],
#     [0.00431511, 0.01973579],
#     [0.02011100, 0.02266763],
# ]
X, y = spiral_data(samples=100, classes=3)


fixed_weights1 = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]])
fixed_biases1 = np.array([[0.1, 0.2, 0.3]])

fixed_weights2 = np.array([[0.01, 0.02, 0.03], [0.04, 0.05, 0.06], [0.07, 0.08, 0.09]])
fixed_biases2 = np.array([[0.1, 0.2, 0.3]])

# dense1 = Layer_Dense(2, 3, fixed_weights1, fixed_biases1)
# dense2 = Layer_Dense(3, 3, fixed_weights2, fixed_biases2)
dense1 = Layer_Dense(2, 3)
dense2 = Layer_Dense(3, 3)

activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
print(activation2.output)
loss = loss_function.calculate(activation2.output, y)
# print("loss:", loss)

# print("Y: ", y)
#
# print("yshape: ", len(X.shape))
