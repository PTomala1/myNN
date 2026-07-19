import numpy as np
import math

class Neuron:
    def __init__(self, weights : [], bias : int, fun : str):
        self.weights = np.array(weights)
        self.bias = bias
        self.fun = fun
        self.activationFuncions = {
            "relu" : self.relu,
            "sigmoid" : self.sigmoid,
        }


    def __str__(self):
        return "Token"

    def __len__(self):
        return len(self.weights)

    def __eq__(self, other):
        return (
                np.array_equal(self.weights, other.weights)
                and self.bias == other.bias
                and self.fun == other.fun
        )

    def __call__(self, x : []):
        return self.run(x)

    def sigmoid(self, Z):
        return 1 / (1 + math.exp(-Z))

    def relu(self, Z):
        return np.maximum(0, Z)

    def run(self, inputs : []):
        inputs = np.array(inputs)

        if len(inputs) != len(self.weights):
            return False

        Z = np.sum(self.weights * inputs) + self.bias
        output = self.activationFuncions[self.fun](Z)
        return output

