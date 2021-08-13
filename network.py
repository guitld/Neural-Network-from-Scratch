import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from layer import Layer

# The network is initialized based on a list of dicts. Each dicts represents each layer
# nn_architecture = [
#     {"input_dim": x, "output_dim": y, "activation": relu/sigmoid},
# ]   
class NeuralNetwork:
    def __init__(self, architecture):
        self.architecture = architecture
        self.num_layers = len(architecture)
        self.layers = []
        for layer in architecture:
            layer_input_size = layer["input_dim"]
            layer_output_size = layer["output_dim"]
            self.layers.append(Layer(layer_input_size, layer_output_size))
            
    # Calls forwardprop method for each layer
    def forward_prop(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # Calls backprop method for each layer
    def back_prop(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    # Training method
    def train(self, train_data, mini_batch_size, n_epochs, learning_rate, test_data=None):
        for epoch in range(n_epochs):
            shuffle(train_data)
            mini_batches = [train_data[i:i+mini_batch_size] for i in range(0, len(train_data), mini_batch_size)]
            loss = 0
            for mini_batch in mini_batches:
                loss = self.update(mini_batch, learning_rate)
            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {len(test_data)}")
            print(f"Epoch {epoch}: {loss:.5f}")

    def update(self, mini_batch, learning_rate):
        loss = 0
        # Forward propagation -> backpropagation -> compute loss
        for x, y in mini_batch:
            output = self.forward_prop(x)
            loss += self.mse(y, output)
            grad = self.mse_grad(y, output)
            self.back_prop(grad, learning_rate)
        return loss / len(mini_batch)

    def evaluate(self, test_data):
        test_results = [(self.prob_to_label(self.forward_prop(x)), self.prob_to_label(y)) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_grad(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)

    def prob_to_label(self, y):
        index = np.argmax(y)
        if index == 9:
            label = 0
        else:
            label = index + 1
        return label