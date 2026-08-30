import math
from shutil import which

from sympy import shape
from torch import FloatTensor
from torchvision import datasets, transforms
import numpy as np
import NN as nn

train = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

# This function takes 2D matix and changes it to be 1D vector
def unsquize(matrix : []):
    r = [] #return var, will contain 1D vector
    for vec in matrix:
        for i in vec:
            r.append(i.tolist())
    return FloatTensor(r)

# This funcction reads output of neural network and returns you
def readNetwork(data : []):
    return data.index(max(data))

# Loss function
def cross_entropy(quess : {}):
    label = readNetwork(quess)
    probability = quess[label]
    loss = -math.log(probability)
    return loss

network = nn.NN()
network.createLayer(784, 128, None, "sigmoid")
network.createLayer(128, 64, None, "sigmoid")
network.createLayer(64, 10, None, "sigmoid")

for i in range(5):
    for N in range(len(train)):
        # Forwoard
        localImg, localLabel = train[N]
        localImg = unsquize(localImg)
        localImg = unsquize(localImg)
        quess = network.forward(localImg)
        # Loss says how wrong NN was
        loss = cross_entropy(quess)
        # Backpropagation calculates which neuron need what correcion
        Y = [0,0,0,0,0,0,0,0,0,0]
        Y[localLabel] = 1
        matrix = network.matixforward(localImg) # It returns exaict result
        dZ3 = matrix[3] - Y                                                         # Output layer error (delta)
        dB3 = dZ3                                                                   # Output layer - 10 biases
        dW3 = np.dot(dZ3.reshape(10,1), matrix[2].reshape(1,64))                    # OutputLayer - 640 neurons
        dZ2 = np.dot(np.transpose(dW3), dZ3) * (matrix[2] * (1.0 - matrix[2]))      # Secound layer error (delta)
        dB2 = dZ2                                                                   # Secound hidden layer - 64 biases
        dW2 = np.dot(dZ2.reshape(64,1), matrix[1].reshape(1, 128))                  # Secound hidden layer - 8192 neurons
        dZ1 = np.dot(np.transpose(dW2), dZ2) * (matrix[1] * (1.0 - matrix[1]))      # Firstlayer error (delta)
        dB1 = dZ1                                                                   # First hidden layer - 128 biases
        dW1 = np.dot(dZ1.reshape(128,1), matrix[0].reshape(1,784))                  # First hidden layer - 100352 neurons



        break
    break