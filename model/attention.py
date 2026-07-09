import torch
import torch.nn as nn
from torchtyping import TensorType

from math import sqrt

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        self.Key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.Value = nn.Linear(embedding_dim, attention_dim, bias=False)
        # Instantiation order matters for reproducible weights: key, query, value
        pass

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        #tensor = self.V(self.Q(self.K(embedded)))
        K = self.Key(embedded)
        Q = self.Query(embedded)
        V = self.Value(embedded)
        print(Q.shape)
        print(K.shape)
        print((K.T).shape)
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        attention = Q @ K.transpose(-2, -1) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        mask = torch.tril(torch.ones(attention.shape))
        masked_scores = attention.masked_fill(mask == 0, float('-inf'))
        # 4. Apply softmax(dim=2) to masked scores
        scores = nn.Softmax(dim=2)(masked_scores)
        # 5. Return (scores @ V) rounded to 4 decimal places
        tensor = torch.matmul(scores, V)
        return torch.round(tensor, decimals=4)
        pass
