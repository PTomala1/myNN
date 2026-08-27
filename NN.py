import numpy as np
import math
import Neuron as Neuron


class NN:
    def __init__(self):
        self.structure = []

    def getLayer(self, layer : int) -> []: #Layers are being counted from 0
        return self.structure[layer]

    def setLayer(self, layer : int, neurons : []): #Layers are being counted from 0
        self.structure[layer] = neurons

    def createLayer(self, inputsPerNeuron : int, amountOfNeurons : int, preset, activationFunction : str = "sigmoid"):
        if preset:
            self.structure.append(preset)
            return
        layer = []
        for i in range(amountOfNeurons):
            limit = np.sqrt(6 / (inputsPerNeuron + 1)) # xavier (glorot) inicialization
            weights = np.random.uniform(-limit, limit, inputsPerNeuron)
            neuron = Neuron.Neuron(weights, 0, activationFunction)
            layer.append(neuron)
        self.structure.append(layer)


    def forward(self, data):
        currData = data
        newData = []
        for layer in self.structure:
            for neuron in layer:
                newData.append(neuron.run(currData))
            currData = newData
            newData = []
        return currData

    def matixforward(self, data):
        matrix = {0:data}
        newData = []
        i = 1
        for layer in self.structure:
            for neuron in layer:
                newData.append(neuron.run(matrix[i - 1]))
            matrix[i] = newData
            newData = []
            i += 1
        matrix[1] = np.array(matrix[1])
        matrix[2] = np.array(matrix[2])
        matrix[3] = np.array(matrix[3])
        return matrix