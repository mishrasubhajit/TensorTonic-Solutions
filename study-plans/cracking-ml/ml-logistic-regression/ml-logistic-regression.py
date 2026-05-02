import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """

    X = np.array(X)
    y = np.array(y)

    n, d = np.shape(X)

    w = np.zeros(d)
    b = 0.0

    def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

    for iter in range(n_iters):
        # forward pass
        z = np.dot(X, w) + b
        y_hat = sigmoid(z)

        
        # gradient descent
        dw = (1.0/n) * (X.T @ (y_hat - y))
        db = (1.0/n) * np.sum(y_hat - y)

        # update the weights
        w = w - lr * dw
        b = b - lr * db

    return (w, b)