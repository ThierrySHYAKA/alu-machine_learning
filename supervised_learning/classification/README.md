0x01-classification
This directory contains work to build up skills needed to be able to build your own binary image classifier from scratch using numpy.

Mandatory Tasks:
Neuron
Write a Python class Neuron that defines a single neuron performing binary classification, including public instance attributes W the weights vector for the neuron, b the bias for the neuron, and A the activated output of the neuron.
Privatize Neuron
Write a Python class Neuron that defines a single neuron performing binary classification, based on 0-neuron.py but all previously public instance attributes become private instance attributes (__W, __b, and __A), each with their own getter function.
Neuron Forward Propagation
Write a Python class Neuron that defines a single neuron performing binary classification, based on 1-neuron.py with the added public method def forward_prop(self, X) that calculates forward propagation of the neuron.
Neuron Cost
Write a Python class Neuron that defines a single neuron performing binary classification, based on 2-neuron.py with the added public method def cost(self, Y, A) that calculates the cost of the model using logistic regression.
Evaluate Neuron
Write a Python class Neuron that defines a single neuron performing binary classification, based on 3-neuron.py with the added public method def evaluate(self, X, Y) that evaluates the neuron's predictions.
Neuron Gradient Descent
Write a Python class Neuron that defines a single neuron performing binary classification, based on 4-neuron.py with the added public method def gradient_descent(self, X, Y, A, alpha=0.05) that calculates one pass of gradient descent on the neuron.
Train Neuron
Write a Python class Neuron that defines a single neuron performing binary classification, based on 5-neuron.py with the added public method def train(self, X, Y, iterations=5000, alpha=0.05) that trains the neuron.
Upgrade Train Neuron
Write a Python class Neuron that defines a single neuron performing binary classification, based on 6-neuron.py that updates the public method train to def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100) that trains the neuron and updates the private attributes __W, __b, and __A.
NeuralNetwork
Write a Python class NeuralNetwork that defines a neural network with one hidden layer performing binary classification, with public instance attributes W1 the weights vector for the hidden layer, b1 the bias for the hidden layer, A1 the activated output for the hidden layer, W2 the weights vector for the output neuron, b2 the bias for the output neuron, and A2 the activated output for the output neuron.
Privatize NeuralNetwork
Write a Python class NeuralNetwork that defines a neural network with one hidden layer performing binary classification, based on 8-nerual_network.py but all previously public instance attributes become private instance attributes (__W1, __b1, __A1, __W2, __b2, and __A2), each with their own getter function.
NeuralNetwork Forward Propagation
Write a Python class NeuralNetwork that defines a neural network with one hidden layer performing binary classification, based on 9-nerual_network.py with the added public method def forward_prop(self, X) that calculates forward propagation of the neural network.
NeuralNetwork Cost
Write a Python class NeuralNetwork that defines a neural network with one hidden layer performing binary classification, based on 10-nerual_network.py with the added public method def cost(self, Y, A) that calculates the cost of the model using logistic regression.
Evaluate NeuralNetwork
Write a Python class NeuralNetwork that defines a neural network with one hidden layer performing binary classification, based on 11-nerual_network.py with the added public method def evaluate(self, X, Y) that evaluates the neural network's predictions.
NeuralNetwork Gradient Descent
Write a Python class NeuralNetwork that defines a neural network with one hidden layer performing binary classification, based on 12-nerual_network.py with added public method def gradient_descent(self, X, Y, A1, A2, alpha=0.05) that calculates one pass of gradient descent on the neural network.
Train NeuralNetwork
Write a Python class NeuralNetwork that defines a neural network with one hidden layer performing binary classification, based on 13-nerual_network.py with the added public method def train(self, X, Y, iterations=5000, alpha = 0.05) that trains the neural network and updates the private attributes __W1, __b1, __A1, __W2, __b2, and __A2.
Upgrade Train NeuralNetwork
Write a Python class NeuralNetwork that defines a neural network with one hidden layer performing binary classification, based on 14-nerual_network.py that updates the public method train to def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100) that trains the neural network and updates the private attributes __W1, __b1, __A1, __W2, __b2, and __A2.
DeepNeuralNetwork
Write a Python class DeepNeuralNetwork that defines a deep neural network performing binary classification, with public instance attributes L the number of layers in the neural network, cache a dictionary to hold all intermediarty values of the network, and weights a dictionary holding all the weights and biases of the network.
Privatize DeepNeuralNetwork
Write a Python class DeepNeuralNetwork that defines a deep neural network performing binary classification, based on 16-deep_nerual_network.py but all previously public instance attributes become private instance attributes (__L, __cache, and __weights), each with their own getter function.
DeepNeuralNetwork Forward Propagation
Write a Python class DeepNeuralNetwork that defines a deep neural network performing binary classification, based on 17-deep_nerual_network.py with the added public method def forward_prop(self, X) that calculates forward propagation of the neural network.
DeepNeuralNetwork Cost
Write a Python class DeepNeuralNetwork that defines a deep neural network performing binary classification, based on 18-deep_nerual_network.py with the added public method def cost(self, Y, A) that calculates the cost of the model using logistic regression.
Evaluate DeepNeuralNetwork
Write a Python class DeepNeuralNetwork that defines a deep neural network performing binary classification, based on 19-deep_nerual_network.py with the added public method def evaluate(self, X, Y) that evaluates the neural network's predictions.
DeepNeuralNetwork Gradient Descent
Write a Python class DeepNeuralNetwork that defines a deep neural network performing binary classification, based on 20-deep_nerual_network.py with added public method def gradient_descent(self, Y, cache, alpha=0.05) that calculates one pass of gradient descent on the neural network.
Train DeepNeuralNetwork
Write a Python class DeepNeuralNetwork that defines a deep neural network performing binary classification, based on 21-deep_nerual_network.py with the added public method def train(self, X, Y, iterations=5000, alpha = 0.05) that trains the neural network and updates the private attributes __weights and __cache.
Upgrade Train DeepNeuralNetwork
Write a Python class DeepNeuralNetwork that defines a deep neural network performing binary classification, based on 22-deep_nerual_network.py that updates the public method train to def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100) that trains the neural network and updates the private attributes __weights and __cache.
One-Hot Encode
Write a function def one_hot_encode(Y, classes) that converts a numeric label vector into a one-hot matrix.
One-Hot Decode
Write a function def one_hot_decode(one_hot) that converts a one-hot matrix into a vector of labels.
Persistence is Key
Write a Python class DeepNeuralNetwork that defines a deep neural network performing binary classification, based on 23-deep_nerual_network.py with added instance method def save(self, filename) that saves the instance object to a file in pickle format and added static method def load(filename) that loads a pickled DeepNeuralNetwork object from a file.
Update DeepNeuralNetwork
Write a Python class DeepNeuralNetwork that updates a deep neural network performing binary classification, based on 26-deep_nerual_network.py to perform multiclass classification, updating the instance methods forward_prop, cost, and evaluate and where Y is now a one-hot numpy.ndarray.
All the Activations
Write a Python class DeepNeuralNetwork that updates a deep neural network performing multiclass classification, based on 27-deep_nerual_network.py to allow different activation functions, updating the __init__ method to also pass in activation function parameter and updates the forward_prop and gradient descent methods to use the specified __activation function in the hidden layers.
test_files directory
The test_files directory contains all files used to test output locally.

data directory
The data directory contains datasets to test the code with.
