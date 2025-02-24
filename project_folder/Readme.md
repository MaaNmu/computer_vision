Overview
This Python script implements a simple two-layer neural network to classify handwritten digits from the MNIST dataset. The network consists of an input layer, a hidden layer with ReLU activation, and an output layer with softmax activation. The script includes functions for initializing parameters, forward propagation, backward propagation, and updating parameters using gradient descent. It also includes utilities for making predictions and visualizing results.

Dependencies
pandas: For reading the dataset.

numpy: For numerical operations.

matplotlib: For visualizing the images.

Data Preparation
The dataset is loaded from data/train.csv. The data is shuffled and split into a training set and a development (dev) set.

Training Set: 41,000 examples.

Development Set: 1,000 examples.

Each example is a 28x28 grayscale image, flattened into a 784-dimensional vector. The pixel values are normalized to the range [0, 1] by dividing by 255.

Neural Network Architecture
Input Layer (A^0): 784 units (one for each pixel).

Hidden Layer (A^1): 10 units with ReLU activation.

Output Layer (A^2): 10 units with softmax activation (one for each digit class).

Functions
init_params()
Initializes the weights and biases for the neural network.

Returns:

W1: Weights for the hidden layer (10x784).

b1: Biases for the hidden layer (10x1).

W2: Weights for the output layer (10x10).

b2: Biases for the output layer (10x1).

ReLU(Z)
Applies the ReLU activation function element-wise.

Parameters:

Z: Input matrix.

Returns: Matrix with ReLU applied.

softmax(Z)
Applies the softmax activation function.

Parameters:

Z: Input matrix.

Returns: Matrix with softmax applied.

forward_prop(W1, b1, W2, b2, X)
Performs forward propagation through the network.

Parameters:

W1, b1, W2, b2: Weights and biases.

X: Input data.

Returns:

Z1, A1, Z2, A2: Intermediate and final layer outputs.

ReLU_deriv(Z)
Computes the derivative of the ReLU function.

Parameters:

Z: Input matrix.

Returns: Matrix with ReLU derivative applied.

one_hot(Y)
Converts labels to one-hot encoded vectors.

Parameters:

Y: Labels.

Returns: One-hot encoded matrix.

backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
Performs backward propagation to compute gradients.

Parameters:

Z1, A1, Z2, A2: Intermediate and final layer outputs.

W1, W2: Weights.

X: Input data.

Y: Labels.

Returns:

dW1, db1, dW2, db2: Gradients for weights and biases.

update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
Updates the weights and biases using gradient descent.

Parameters:

W1, b1, W2, b2: Current weights and biases.

dW1, db1, dW2, db2: Gradients.

alpha: Learning rate.

Returns: Updated weights and biases.

get_predictions(A2)
Gets the predicted class labels from the output layer.

Parameters:

A2: Output layer activations.

Returns: Predicted labels.

get_accuracy(predictions, Y)
Computes the accuracy of the predictions.

Parameters:

predictions: Predicted labels.

Y: True labels.

Returns: Accuracy as a float.

gradient_descent(X, Y, alpha, iterations)
Performs gradient descent to train the network.

Parameters:

X: Input data.

Y: Labels.

alpha: Learning rate.

iterations: Number of iterations.

Returns: Trained weights and biases.

make_predictions(X, W1, b1, W2, b2)
Makes predictions using the trained network.

Parameters:

X: Input data.

W1, b1, W2, b2: Trained weights and biases.

Returns: Predicted labels.

test_prediction(index, W1, b1, W2, b2)
Tests a single prediction and visualizes the image.

Parameters:

index: Index of the example to test.

W1, b1, W2, b2: Trained weights and biases.

Usage
Training: The network is trained using the gradient_descent function.

Testing: Predictions are made on the development set using make_predictions.

Visualization: The test_prediction function is used to visualize individual predictions.

Example
python
Copy
# Train the network
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.1, 500)

# Test predictions
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)

# Evaluate on the dev set
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
accuracy = get_accuracy(dev_predictions, Y_dev)
print(f"Accuracy on the testing set: {accuracy}")
Notes
The script assumes the dataset is in data/train.csv.

The learning rate and number of iterations can be adjusted for better performance.

The script includes detailed comments for understanding each step of the process.
