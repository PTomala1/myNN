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
    vocab = {}
    for f in data:
        vocab[len(vocab)] = f
    r = max(vocab.values(), key=vocab.keys())
    print(vocab)
    print(r)
    # return vocab

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
        print(readNetwork(quess))
        break
    break