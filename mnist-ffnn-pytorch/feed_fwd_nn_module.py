import torch
from torch import nn
import torch.nn.functional as F

from typing import List
from consts import Consts

class FeedFwdNNModule(nn.Module):
    def __init__(
        self
        , input_size    : int
        , output_size   : int
        , dropout_prob  : float
        , hidden_sizes  : List[int]
    ):
        super(FeedFwdNNModule, self).__init__()

        sizes = [input_size] + hidden_sizes + [output_size]
        
        self.L = nn.ModuleList(
            [nn.Linear(sizes[idx], sizes[idx+1]) for idx in range(len(sizes) - 1)]
        )
        
        self.D = nn.ModuleList(
            [nn.Dropout(p=dropout_prob) for _ in range(len(hidden_sizes))]
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X.view(X.size(0), -1)
        
        for l_idx, l in enumerate(self.L[:-1]):
            X = F.relu(l(X))
            
            if l_idx < len(self.D):
                X = self.D[l_idx](X)
                
        X = self.L[-1](X)
        
        return X
    