import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X, y = np.array(X, dtype=float), np.array(y, dtype=float)
    n, d = X.shape

    w = np.zeros(d)
    b = 0.0

    for iter in range(steps):
        z = X @ w + b
        y_pred = _sigmoid(z)
        error = y_pred - y

        dw = (1.0/n) *  (X.T @ error)
        db = np.mean(error)

        w = w - lr * dw
        b = b - lr * db

    return (w, b)
    pass