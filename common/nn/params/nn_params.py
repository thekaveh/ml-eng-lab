from __future__ import annotations

import ast

from typing import List, Optional
from dataclasses import dataclass, field

from ..enum.activations import Activations

@dataclass(frozen=True, kw_only=True, slots=True)
class NNParams:
    dropout_prob    : float
    input_dim       : int                       = field(repr=False)
    output_dim      : int                       = field(repr=False)
    hidden_dims     : Optional[List[int]]       = field(repr=False, default=None)
    activation      : Optional[Activations]     = field(repr=False, default=Activations.LEAKY_RELU)
    
    _dims           : Optional[List[int]]       = field(repr=False, init=False, default=None)
    
    @property
    def dims(self) -> List[int]:
        return self._dims
    
    def __post_init__(self):
        dims = [self.input_dim]
        dims += self.hidden_dims if self.hidden_dims is not None else []
        dims += [self.output_dim]
        
        object.__setattr__(self, '_dims', dims)
    
    def __repr__(self):
        return f"{{dims={self.dims}, activation={self.activation}, dropout={self.dropout_prob:0.2f}}}"
    
    def __str__(self):
        return self.__repr__()
    
    def to_dict(self) -> dict:
        return dict(
            input_dim       = self.input_dim
            , output_dim    = self.output_dim
            , dropout_prob  = self.dropout_prob
            , hidden_dims   = str(self.hidden_dims)
            , activation    = str(self.activation)
        )
    
    @staticmethod
    def from_dict(rep: dict) -> NNParams:
        return NNParams(
            input_dim       = rep['input_dim']
            , output_dim    = rep['output_dim']
            , dropout_prob  = rep['dropout_prob']
            , hidden_dims   = ast.literal_eval(rep['hidden_dims'])
            , activation    = Activations(rep['activation'])
        )

@dataclass(frozen=True, kw_only=True, slots=True)
class GraphAttNNParams(NNParams):
    n_heads: int = field(repr=True, init=True, default=4)
    
    def __repr__(self):
        return f"{{dims={self.dims}, dropout={self.dropout_prob:0.2f}, heads={self.n_heads}}}"
    
    def __str__(self):
        return self.__repr__()
    
    def to_dict(self) -> dict:
        return dict(
            input_dim       = self.input_dim
            , output_dim    = self.output_dim
            , dropout_prob  = self.dropout_prob
            , hidden_dims   = str(self.hidden_dims)
            , activation    = str(self.activation)
            , n_heads         = self.n_heads
        )
    
    @staticmethod
    def from_dict(rep: dict) -> GraphAttNNParams:
        return GraphAttNNParams(
            input_dim       = rep['input_dim']
            , output_dim    = rep['output_dim']
            , dropout_prob  = rep['dropout_prob']
            , hidden_dims   = ast.literal_eval(rep['hidden_dims'])
            , activation    = Activations(rep['activation'])
            , n_heads       = rep['n_heads']
        )