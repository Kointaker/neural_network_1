import numpy as np

def sigmoid(x):
    # The activation function: f(x) = 1 / (1 + e^(-x)) 
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # Weight inputs, add bias, then use the activation function
        # Calculates this by finding sum of element-wise products
        # [1, 2, 3]
        # [4, 5, 6]
        # This would be 1*4, 2*5, and 3*6 respectively
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)
    
weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
n = Neuron(weights, bias) # our neuron now with desired weights and bias

x = np.array([2, 3])       # x1 = 2, x2 = 3
print(n.feedforward(x))    # 0.9990889488055994

