import numpy as np

def sigmoid(x):
    # The activation function: f(x) = 1 / (1 + e^(-x)) 
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.bias = bias

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        # Calculates this by finding sum of element-wise products
        # [1, 2, 3]
        # [4, 5, 6]
        self.weights = weights
        # This would be 1*4, 2*5, and 3*6 respectively
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
# weights and bias set outside of class
# to allow neuron to function with various
# inputs/neural networks
weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias) # our neuron now with desired weights and bias
x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x))    # 0.9990889488055994



class OurNeuralNetwork:
    '''

    A neural network with:
        - 2 inputs
        - a hidden layer with 2 neurons (h1, h2)
        - an output layer with 1 neuron (o1)
    Each neuron has the same weights and bias:
        - w = [0, 1]
        - b = 0
    '''

    def __init__(self):
        # weights and bias set for this neural network
        # each neuron will now have these parameters
        # when calculating output
        weights = np.array([0, 1])
        bias = 0

        # Activated neurons in this network with parameters
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)
    # Function inside OurNeuralNetwork
    def feedforward(self, x):
        # Both h1 & h2 neurons get two ff inputs
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        # o1 neuron gets two neuron outputs as inputs
        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

        return out_o1

# Creating the neuralnetwork
network = OurNeuralNetwork()
x = np.array([2, 3])

# Essentially, h1 and h2 are getting the x as their inputs, then 
# o1 gets this outputs as it's input, and the output
# of the network is the output of o1
print(network.feedforward(x)) # running function in created neural network: 0.7216325609518421
