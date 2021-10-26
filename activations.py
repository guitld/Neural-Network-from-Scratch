import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def softmax_grad(x):
    return x

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) - np.exp(-x))

def tanh_grad(x):
    return 1 - np.power(tanh(x), 2)

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_grad(x):
    return 1 / 1 + np.exp(-x)

def gaussian(x):
    return np.exp(-np.power(x, 2))

def gaussian_grad(x):
    return -2 * np.exp(-np.power(x, 2))

def sigmoid(x):
    s = 1/(1+np.exp(-x)) 
    return s

def sigmoid_grad(x):
    s = sigmoid(x)
    ds = s*(1-s)
    return ds

activation_functions = {
    "relu": [relu, relu_grad],
    "softmax": [softmax, softmax_grad],
    "tanh": [tanh, tanh_grad],
    "softplus": [softplus, softplus_grad],
    "gaussian": [gaussian, gaussian_grad],
    "sigmoid": [sigmoid, sigmoid_grad]
}