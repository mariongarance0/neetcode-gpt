import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
        self.first_layer = nn.Linear(784, 512)
        self.relu_layer = nn.ReLU()
        self.Dropout_layer = nn.Dropout(0.2)
        self.linear_layer = nn.Linear(512, 10)
        self.sigmoid_layer = nn.Sigmoid()
        
        pass

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        batch_size = images.shape[0]
        tensor = self.sigmoid_layer(self.linear_layer(self.Dropout_layer(self.relu_layer(self.first_layer(images)))))
        tensor = torch.round(tensor, decimals=4)
        return tensor 
        # Return the model's prediction to 4 decimal places
        pass
