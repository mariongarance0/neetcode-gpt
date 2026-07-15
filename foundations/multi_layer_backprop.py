import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        W1 = np.array(W1)
        W2 = np.array(W2)
        b1 = np.array(b1)
        b2 = np.array(b2)
        y_true = np.array(y_true)
        x = np.array(x)

        z1 = x @ np.transpose(W1) + b1

        a1 = np.maximum(0,z1)
        z2 = a1 @ np.transpose(W2) + b2

        loss = np.mean((z2 - y_true)**2)

        dz2 = 2*(z2 - y_true)/len(y_true)
        db2 = dz2

        dW2 = dz2.reshape(-1, 1) @ a1.reshape(1, -1)
        print( np.where(z1 > 0, 1, 0))
        print(dz2.reshape(-1, 1) @ W2.reshape(1, -1))
        db1 = dz2.reshape(-1, 1) @ W2.reshape(1, -1) * np.where(z1 > 0, 1, 0)

        db1 = db1.flatten()

        dW1 = db1.reshape(-1, 1) @ x.reshape(1, -1)

        return {
            'loss': np.round(loss,4),  
            'dW1': np.round(dW1,4), 
            'db1': np.round(db1,4), 
            'dW2': np.round(dW2,4), 
            'db2': np.round(db2,4)
            }
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        pass
