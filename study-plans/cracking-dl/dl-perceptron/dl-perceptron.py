import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n, d = X.shape
    weights = np.zeros(d)
    bias = 0.0

    def activation(real):
        """step function"""
        return 1 if real >= 0 else 0

    for epoch in range(epochs):
        for idx, inputs in enumerate(X):
            z = np.dot(weights, inputs) + bias
            y_pred = activation(z)
            error = y[idx] - y_pred
            weights += lr * error * inputs
            bias += lr * error
    return weights.tolist(), bias