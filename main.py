import math
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

def readNetwork(data : []):
    return data.index(max(data))

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
        localImg, localLabel = train[N]
        localImg = unsquize(localImg)
        localImg = unsquize(localImg)
        quess = network.forward(localImg)
        loss = cross_entropy(quess)
        Y = [0,0,0,0,0,0,0,0,0,0]
        Y[localLabel] = 1
        matrix = network.matixforward(localImg)
        # print(matrix)

        dZ3 = matrix[3] - Y
        dB3 = dZ3
        # dW3 = np.dot(dZ3, np.transpose(matrix[2]))
        # print(dW3)
        break
    break