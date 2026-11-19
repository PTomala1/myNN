import math
from shutil import which

from matplotlib import pyplot as plt
from sympy import shape
from torch import FloatTensor
from torchvision import datasets, transforms
import numpy as np
import NN as nn

train = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())
test = datasets.MNIST(root="data", train=False, download=True, transform=transforms.ToTensor())

batchs_train = []
batch_size = 32


# for multiplier in range(len(train)//32):
#     batchs_train[multiplier].append([])
#     for i in range(32):
#         batchs_train[multiplier].append(train[ (multiplier * 32)+i])

for i in range(0, len(train), batch_size):
    batch = [train[j] for j in range(i, min(i + batch_size, len(train)))]
    batchs_train.append(batch)


def unsquize(matrix : list):
    r = [] #return var
    for vec in matrix:
        for i in vec:
            r.append(i.tolist())
    return FloatTensor(r)

# This funcction reads output of neural network and returns you
def readNetwork(data : list):
    return np.argmax(data)

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

def winLossRatio():

    won = 0
    for img, label in test:
        img = unsquize(img)
        img = unsquize(img)
        quess = readNetwork(network.forward(img))
        if (quess == label):
            won += 1
    return won/len(test)

forPLT = [(x+1) for x in range(25)]

def pltShow(X,Y):
    plt.xlabel("Oś Epoka")
    plt.ylabel("Oś % poprawności * 10")
    plt.plot(X, Y)
    plt.grid(True)
    plt.show()

#training
learning_rate = 0.01
lossPerEpok = []
for i in range(20):
    for curr_batch in batchs_train:
        dW3_total, dB3_total = 0, 0
        dW2_total, dB2_total = 0, 0
        dW1_total, dB1_total = 0, 0
        for img, label in curr_batch:
            # Forwoard
            img = unsquize(img)
            img = unsquize(img)
            matrix = network.matixforward(img)  # It returns exaict result of each layer





            # Backpropagation calculates which neuron need what correcion
            Y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            Y[label] = 1
            dZ3 = matrix[3] - Y  # Output layer error (delta)
            dB3 = dZ3  # Output layer - 10 biases
            dW3 = np.dot(dZ3.reshape(10, 1), matrix[2].reshape(1, 64))  # OutputLayer - 640 neurons
            W3 = np.array(network.getWeights(2))
            dZ2 = np.dot(np.transpose(W3), dZ3) * (matrix[2] * (1.0 - matrix[2]))  # Secound layer error (delta)
            dB2 = dZ2  # Secound hidden layer - 64 biases
            dW2 = np.dot(dZ2.reshape(64, 1), matrix[1].reshape(1, 128))  # Secound hidden layer - 8192 neurons
            W2 = np.array(network.getWeights(1))
            dZ1 = np.dot(np.transpose(W2), dZ2) * (matrix[1] * (1.0 - matrix[1]))  # Firstlayer error (delta)
            dB1 = dZ1  # First hidden layer - 128 biases
            dW1 = np.dot(dZ1.reshape(128, 1), matrix[0].reshape(1, 784))  # First hidden layer - 100352 neurons

            #Gradient
            dW3_total += dW3
            dB3_total += dB3
            dW2_total += dW2
            dB2_total += dB2
            dW1_total += dW1
            dB1_total += dB1

        dB3_total /= batch_size
        dW3_total /= batch_size
        dW2_total /= batch_size
        dB2_total /= batch_size
        dW1_total /= batch_size
        dB1_total /= batch_size

        #Gradient descent
        network.update_weights(2, dW3_total, dB3_total, learning_rate)
        network.update_weights(1, dW2_total, dB2_total, learning_rate)
        network.update_weights(0, dW1_total, dB1_total, learning_rate)
    lossPerEpok.append(winLossRatio() * 10)
    pltShow(forPLT, lossPerEpok)









# print(testLabel)
#
# print(quess)
# img_to_show = testImg.squeeze(0)
#
# # Wyświetlanie obrazu
# plt.imshow(img_to_show, cmap='gray')
# plt.title(f'Prawidłowa etykieta (Cyfra): {testLabel}')
# plt.axis('off')  # Wyłącza osie z współrzędnymi
# plt.show()