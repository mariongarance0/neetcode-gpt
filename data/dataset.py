import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        torch.manual_seed(0)
        # 1. Tokenize by splitting on whitespace: 
        tokens = raw_dataset.split()
        # 2. Generate batch_size random start indices using 
        indices = torch.randint(0, len(tokens) - context_length, size=(batch_size,)).tolist()
        #    Range: [0, len(tokens) - context_length)
        # 3. 
        X = []
        Y = []
        for i in indices:
            X.append(tokens[i:i+context_length])
            Y.append(tokens[i+1:i+1+context_length])
        
        return (X, Y)
        pass
