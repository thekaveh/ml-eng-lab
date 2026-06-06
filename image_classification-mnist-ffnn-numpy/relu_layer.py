import numpy as np

from funcs import Funcs
from consts import Consts

class ReluLayer:
    def __init__(
        self
        , feature_size: int = Consts.FEATURES_SIZE_OUT
    ):
        assert feature_size is not None and feature_size > 0

        self.feature_size = feature_size

    # Z: nxc -> A: nxc
    def forward(self, Z: np.matrix):
        assert Z is not None
        assert Z.ndim == 2
        assert Z.shape[1] == self.feature_size

        self.Z = Z

        A = Funcs.parametric_relu(Z, Consts.PARAMETRIC_RELU_ALPHA)

        assert A is not None
        assert A.ndim == 2
        assert A.shape[0] == Z.shape[0]
        assert A.shape[1] == self.feature_size

        return A

    # dL_dA: nxc -> dL_dZ: nxc
    def backward(self, dL_dA: np.matrix):
        assert dL_dA is not None
        assert dL_dA.ndim == 2
        assert dL_dA.shape[0] == self.Z.shape[0]
        assert dL_dA.shape[1] == self.feature_size

        dA_dZ = Funcs.parametric_relu_prime(self.Z, Consts.PARAMETRIC_RELU_ALPHA)

        assert dA_dZ is not None
        assert dA_dZ.ndim == 2
        assert dA_dZ.shape[0] == self.Z.shape[0]
        assert dA_dZ.shape[1] == self.feature_size

        dL_dZ = dL_dA * dA_dZ

        assert dL_dZ is not None
        assert dL_dZ.ndim == 2
        assert dL_dZ.shape[0] == self.Z.shape[0]
        assert dL_dZ.shape[1] == self.feature_size

        return dL_dZ
