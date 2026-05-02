import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """

    # converting to numpy arrays
    X = np.array(X)
    y = np.array(y)
    n, d = np.shape(X)

    # initialize weights and bias
    w = np.zeros(d)
    b = 0.0

    for iter in range(epochs):
        y_hat = np.dot(X, w) + b

        dw = (2/n) * np.dot(X.T, y_hat - y)
        db = (2/n) * np.sum(y_hat - y)

        w = w - lr * dw
        b = b - lr * db

    return(w, b)
