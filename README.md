# Neural Network from Scratch

# MNIST Dataset

The MNIST is a dataset of handwritten digits with a training set of 60.0000 samples and a test set of 10.000 samples.
To preprocess the data, all the images were normalized and the output were encoded to vectors

<img src="https://user-images.githubusercontent.com/67521354/129427174-cfbf742e-3708-4944-9676-7102cb57cbc9.png" width="540" height="240">


# The Network's Architecture

The network is a simple multilayer perceptron with one hidden layer. The input layer has 784 (28 * 28 images) input neurons, the hidden layer has 30 neurons and the output has 10. Each neuron of the output represents the probability of a certain digit to be recognized. The network's architecture is defined as a list of dictionaries. Each item represents a layer.

Example: {"input_dim": x, "output_dim": y, "activation": relu/sigmoid} The supported activations functions are ReLU and Sigmoid.

For this graphical representation, the input layer has been shrunk.
<img src="https://i.imgur.com/5A9ikWo.png" width="700" height="640">

