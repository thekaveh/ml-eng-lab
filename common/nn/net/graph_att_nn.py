import torch
import torch_geometric as pyg
import torch.nn.functional as F

from dataclasses import dataclass, field

from torch import nn
from .nn_params import NNParams
from .nn_activation_fn import NNActivationFn

@dataclass(frozen=True, kw_only=True, slots=True)
class GraphAttNNParams(NNParams):
    n_heads: int = field(repr=True, init=True, default=1)
    
    def __repr__(self):
        return f"{{dims={self.dims}, dropout={self.dropout_prob:0.2f}, heads={self.n_heads}}}"
    
    def __str__(self):
        return self.__repr__()

class GraphAttNN(nn.Module):
    def __init__(self, params: GraphAttNNParams):
        super(GraphAttNN, self).__init__()
        
        self.params = params
        
        dim_pairs = list(zip(self.params.dims, self.params.dims[1:]))
        
        self.layers = nn.ModuleList(
            [
                pyg.nn.GATConv(
                    out_channels=out_dim
                    , heads=self.params.n_heads
                    , concat=True if (idx_dim != len(dim_pairs) - 1) else False
                    , in_channels=in_dim if idx_dim == 0 else in_dim * self.params.n_heads
                )   for idx_dim, (in_dim, out_dim) in enumerate(dim_pairs)
            ]
        )

    def forward(self, X: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        for layer in self.layers[:-1]:
            X = layer(X, E)
            X = self.params.activation_fn()(X)
            X = F.dropout(X, p=self.params.dropout_prob, training=self.training)
                
        X = self.layers[-1](X, E)
        
        return X
    
    def unpack_batch(self, batch):
        X, E, Y = batch.x, batch.edge_index, batch.y
        
        return (X, E), Y
    
    def __str__(self):
        return f"GraphAttNN={self.params}"