import torch
import torch.nn as nn
import math
from typing import List
import numpy as np


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = math.sqrt(2/(fan_in + fan_out))
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals=4).tolist()
        pass

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        std = np.sqrt(2/fan_in)
        return torch.round(torch.randn(fan_out, fan_in) * std, decimals=4).tolist()
        
        pass

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.

        dim = [input_dim] + [hidden_dim] * num_layers

        torch.manual_seed(0)
        weights = []
        for i in range(num_layers):
            if init_type == 'xavier':
                std = math.sqrt(2.0 / (dim[i] + dim[i + 1]))
            elif init_type == 'kaiming':
                std = math.sqrt(2.0 / dim[i])
            else:
                std = 1.0
            w = torch.randn(dim[i + 1], dim[i]) * std
            weights.append(w)

        x = torch.randn(1, input_dim)

        std =[]
        for w in weights:
            x = x @ w.T
            x = nn.ReLU()(x)
            std.append(round(torch.std(x).item(), 2))
        return std
        pass
