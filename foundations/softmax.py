import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        div = np.sum(np.exp(z - np.max(z)))
        #result = []
        #for i in range(len(z)):
        #    result.append(np.exp(z[i] - np.max(z))/div)
        return np.round(np.exp(z - np.max(z))/div, 4) 
        pass
