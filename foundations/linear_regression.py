import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        return np.round(np.dot(X, weights), 5)
        pass

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        #diff = model_prediction - ground_truth
        #diffT = np.matrix_transpose(diff)
        (N,C) = model_prediction.shape
        MSE = 0
        for i in range(N):
            MSE += (model_prediction[i] - ground_truth[i])**2
        
        return np.round(MSE[0]/N, 5)
        pass
