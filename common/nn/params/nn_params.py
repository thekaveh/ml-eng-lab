from typing import List, Optional
from dataclasses import dataclass, field

from ..enum.activation_fn import ActivationFn

@dataclass(frozen=True, kw_only=True, slots=True)
class NNParams:
    dropout_prob    : float
    input_dim       : int                       = field(repr=False)
    output_dim      : int                       = field(repr=False)
    hidden_dims     : Optional[List[int]]       = field(repr=False, default=None)
    activation_fn   : Optional[ActivationFn]    = field(repr=False, default=ActivationFn.LEAKY_RELU)
    
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
        return f"{{dims={self.dims}, activation={self.activation_fn}, dropout={self.dropout_prob:0.2f}}}"
    
    def __str__(self):
        return self.__repr__()