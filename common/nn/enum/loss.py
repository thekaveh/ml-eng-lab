from __future__ import annotations

from torch import nn
from enum import Enum
    
class Loss(Enum):
    CROSS_ENTROPY           = "cross_entropy"
    MEAN_SQUARED_ERROR      = "mean_squared_error"
    BINARY_CROSS_ENTROPY    = "binary_cross_entropy"
    NEGATIVE_LOG_LIKELIHOOD = "negative_log_likelihood"

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return str(self)

    def __call__(self):
        match self:
            case Loss.CROSS_ENTROPY             : return nn.CrossEntropyLoss()
            case Loss.MEAN_SQUARED_ERROR        : return nn.BCELoss()
            case Loss.BINARY_CROSS_ENTROPY      : return nn.MSELoss()
            case Loss.NEGATIVE_LOG_LIKELIHOOD   : return nn.NLLLoss()