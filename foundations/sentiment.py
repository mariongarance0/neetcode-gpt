import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)

        if vocabulary_size < 1:
            raise ValueError("vocabulary_size should be equal or greater than 1")

        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embedding = nn.Embedding(vocabulary_size,16)
        self.linear = nn.Linear(16,1)
        self.sig = nn.Sigmoid()
        pass

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        embedded = self.embedding(x)
        mean = torch.mean(embedded, 1)
        lin = self.linear(mean)
        return (self.sig(lin))
        # Return a B, 1 tensor and round to 4 decimal places
        pass
