import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        mean_feat = np.mean(x, axis = 0)
        var_feat = np.var(x, axis = 0)
        
        if training == True:
            running_mean=np.array([(1-momentum)*r_mean for r_mean in running_mean])+np.array([momentum * mean for mean in mean_feat])
            running_var = np.array([(1-momentum)*r_mean for r_mean in running_var])+np.array([momentum * mean for mean in var_feat])
            x_hat = (x-mean_feat)/(np.sqrt(var_feat+eps))
            y = gamma * x_hat + beta

        if training == False:
            x_hat = (np.array(x)-np.array(running_mean))/(np.sqrt([rv+eps for rv in running_var]))
            y = gamma * x_hat + beta
            

        
        return (np.round(y, 4), np.round(running_mean, 4), np.round(running_var,4))
        pass
