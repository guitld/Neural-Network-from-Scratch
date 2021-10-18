import numpy as np
import activations

class Layer:
    def __init__(self, input_dim, output_dim, activation_func="sigmoid"):
        self.weights = np.random.randn(output_dim, input_dim)
        self.biases = np.random.randn(output_dim, 1)
        
        self.activation = activations.activation_functions[activation_func][0]
        self.activation_grad = activations.activation_functions[activation_func][1]

    # Calculates the next activation layer
    def forward(self, x):
        self.input = x
        self.z = np.dot(self.weights, self.input) + self.biases
        return self.activation(self.z)

    # Updates the weights and biases based on the chain rule
    #  dL    dL   dA   dZ       |  dL                  |  dA                        |  dZ            |
    # ———— = —— · —— · ——  ---> |  —— = dLossFunction  |  —— = dActivationFunction  |  —— = A_(i-1)  |
    # dW_i   dA   dZ   dW       |  dA                  |  dZ                        |  dW            |
    # ------------------------------------------------------------------------------------------------
    #  dL    dL   dA   dZ       |  dL                  |  dA                        |  dZ            |
    # ———— = —— · —— · ——  ---> |  —— = dLossFunction  |  —— = dActivationFunction  |  —— = 1        |
    # dB_i   dA   dZ   dW       |  dA                  |  dZ                        |  dB            |
    
    def backward(self, output_grad, learning_rate):
        output_grad = np.multiply(output_grad, self.activation_grad(self.z))
        dW = np.dot(output_grad, self.input.transpose())
        self.weights = self.weights - learning_rate * dW
        self.biases -= self.biases - learning_rate * output_grad
        return np.dot(self.weights.transpose(), output_grad) 