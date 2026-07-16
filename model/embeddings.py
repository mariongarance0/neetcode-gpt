import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Return the embedding vectors for the given token IDs
        # return np.round(your_answer, 5)
        (vocab_size, embed_dim) = np.shape(np.array(embeddings))
        out = np.zeros((len(token_ids), embed_dim))
        for i in range(len(token_ids)):
            out[i] = embeddings[token_ids[i]]
        return np.round(out, 5)
        pass
