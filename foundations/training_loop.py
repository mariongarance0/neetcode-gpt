import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))
        n_features = X.shape[1]
        n_samples = X.shape[0]
        w: npt.NDArray[np.float64] = np.zeros(n_features, dtype=np.float64)
        b: float = 0.0
        for i in range(epochs):
            y_hat: npt.NDArray[np.float64] = X @ w + b
            dw = 2/n_samples * np.array(X.T @ (y_hat - y))
            db = 2/n_samples * np.sum(np.array(y_hat-y))
            w = np.array(w) - lr*np.array(dw)
            b = b - lr*db
        return (np.round(w, 5), np.round(b, 5))
        pass
