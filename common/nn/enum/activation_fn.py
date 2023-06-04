from __future__ import annotations

import torch.nn.functional as F

from enum import Enum

class ActivationFn(Enum):
    ELU         = 'elu'
    SELU        = 'selu'
    TANH        = 'tanh'
    RELU        = 'relu'
    SOFTMAX     = 'softmax'
    SIGMOID     = 'sigmoid'
    SOFTPLUS    = 'softplus'
    LEAKY_RELU  = 'leaky_relu'

    def __str__(self):
        return self.value
    
    def __repr__(self):
        return str(self)

    def __call__(self):
        match self:
            case ActivationFn.ELU           : return F.elu
            case ActivationFn.SELU          : return F.selu
            case ActivationFn.TANH          : return F.tanh
            case ActivationFn.RELU          : return F.relu
            case ActivationFn.SOFTMAX       : return F.softmax
            case ActivationFn.SIGMOID       : return F.sigmoid
            case ActivationFn.SOFTPLUS      : return F.softplus
            case ActivationFn.LEAKY_RELU    : return F.leaky_relu