import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=lr
        )       

        for epoch in range(epochs):
            torch.manual_seed(epoch)
            start_list = torch.randint(len(data) - context_length, (batch_size,))
            X = torch.stack([data[start : start + context_length] for start in start_list])
            targets = torch.stack([data[start + 1 : start + 1 + context_length] for start in start_list])
            
            logits = model(X)
            (B,T,C) = logits.shape
            logits_flat = torch.reshape(logits, (B*T, C))
            targets_flat = torch.reshape(targets, (B*T,))

            loss = F.cross_entropy(logits_flat, targets_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)
        pass

"""
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Clear gradients
                outputs = model(inputs)  # Forward pass
                loss = F.cross_entropy(logits_flat, targets_flat)
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights
                running_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
            """
