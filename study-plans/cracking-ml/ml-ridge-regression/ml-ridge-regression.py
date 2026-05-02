def ridge_regression(X, y, lr, epochs, alpha):
    """
    Perform ridge regression using gradient descent.
    Returns: tuple of (weights_list, bias)
    """

    X, y = np.array(X, dtype=float), np.array(y, dtype=float)
    n, d = X.shape

    w = np.zeros(d)
    b = 0.0

    for iter in range(epochs):
        y_pred = X @ w + b
        error = y_pred - y

        # dw_lin_reg = (2.0/n) * (X.T @ error)
        dw = (2.0/n) * (X.T @ error) + 2 * alpha * w
        db = (2.0/n) * np.sum(error)

        w = w - lr*dw
        b = b - lr*db

    weights = [round(float(i), 4) for i in w]
    bias = round(float(b), 4)
    return (weights, bias)
    pass