import numpy as np


def sigmoid(x, derivative=False):
    return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))


class NeuronLayer:
    def __init__(self, size, activation=sigmoid):
        self.size = 0
        self.activation = sigmoid
        self.vector = np.zeros(size)
        self.set(size, activation)

    def zero(self):
        self.vector = np.zeros(self.size)

    def activate(self):
        self.activation(self.vector)

    def set(self, size, activation=sigmoid):
        self.size = size
        self.activation = activation
        self.zero()


class NeuralNetwork:
    def __init__(self, shape, activation=sigmoid, values=None):
        self.coefs = []
        self.layers = []
        self.values = []

        if shape is not None and values is None:
            self.randomize(shape, activation)

    def matrices_from_values(self, values=None):
        if values is not None:
            self.values = values
            self.coefs = []
            j = 0
            for i in range(len(self.layers) - 1):
                matrix = np.zeros((self.layers[i].size, self.layers[i + 1].size))

                for x in range(self.layers[i].size):
                    for y in range(self.layers[i + 1].size):
                        matrix[x, y] = self.values[j]
                        j += 1
                self.coefs.append(np.matrix(matrix))

    def zero(self, shape, activation):
        # Initialize each layer with zeros
        self.layers = []

        for i in range(len(shape)):
            self.layers.append(NeuronLayer(shape[i], activation))

        # Initialize values of transfers matrix in self. Coefs with a linear array
        self.values = []

        for i in range(len(shape) - 1):
            self.values = self.values + [0.0] * shape[i] * shape[i + 1]

        self.values = np.array(self.values)

    def randomize(self, shape, activation):
        self.zero(shape, activation)

        if self.values is not None:
            self.values.rand(len(self.values))

        self.matrices_from_values(self.values)

    def output(self, input_layer):
        if input_layer.size == self.layers[0].size:
            self.layers[0] = input_layer
            output = self.layers[0]

            for i in range(len(self.coefs)):
                output = self.coefs[i] * output

        return output


testLayer = NeuronLayer(6)

testNetwork = NeuralNetwork([3, 2, 3])

# Test program

print("Checking linear values of coefs :")
print(testNetwork.values)

print("Checking matrical values of coefs :")
for coefsMatrix in testNetwork.coefs:
    print(coefsMatrix)

print("Checking layers values :")
for layer in testNetwork.layers:
    print(layer.vector)
