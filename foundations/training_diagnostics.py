import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        out = []
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, nn.Linear):
                    x = layer(x)
                    mean = torch.mean(x).item()
                    std = torch.std(x).item()
                    tot = 0

                    if x.dim() >= 2:
                        dead_frac = round(((x <= 0).all(dim=0)).float().mean().item(), 4)
                    else:
                        dead_frac = round((x <= 0).float().mean().item(), 4)

                    out.append({'mean': round(mean,4), 'std': round(std,4), 'dead_fraction': dead_frac})
                else:
                    x = layer(x)
        return out
        pass

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()
        x = model(x)
        loss = nn.MSELoss()(x, y)
        loss.backward()
        out = []
        with torch.no_grad():
            for layer in model.children():
                if isinstance(layer, nn.Linear):
       
                    grad = layer.weight.grad

                    std = torch.std(grad).item()
                    mean = torch.mean(grad).item()
                    norm = torch.norm(grad).item()

                    out.append({'mean': round(mean,4), 'std': round(std,4), 'norm': round(norm, 4)})
        return out
        pass

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        for i in range(len(activation_stats)):
            if activation_stats[i].get('dead_fraction') > 0.5:
                return 'dead_neurons'

        for i in range(len(gradient_stats)):
            if gradient_stats[i].get('norm') > 1000:
                return 'exploding_gradients'

        if gradient_stats[-1].get('norm') < 1e-5:
            return 'vanishing_gradients'

        for i in range(len(activation_stats)):
            if activation_stats[i].get('std') < 0.1:
                return 'vanishing_gradients'
            if activation_stats[i].get('std') > 10.0:
                return 'exploding_gradients'
        return 'healthy'

        pass
