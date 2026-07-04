import numpy as np

from funcs import softmax, cross_entropy, smce_prime
from consts import Consts

class SoftmaxCrossEntropyLayer:
    def __init__(
        self
        , feature_size: int = Consts.FEATURES_SIZE_OUT
    ):
        assert feature_size is not None and feature_size > 0

        self.feature_size = feature_size

    # A: nxc -> Y_hat: nxc, (Y: nxc, Y_hat: nxc) -> L
    def forward(self, A: np.ndarray, Y: np.ndarray):
        assert A is not None
        assert A.ndim == 2
        assert A.shape[1] == self.feature_size

        assert Y is not None
        assert Y.ndim == 2
        assert Y.shape[0] == A.shape[0]
        assert Y.shape[1] == self.feature_size

        self.Y = Y

        Y_hat = softmax(A)

        assert Y_hat is not None
        assert Y_hat.ndim == 2
        assert Y_hat.shape[0] == A.shape[0]
        assert Y_hat.shape[1] == self.feature_size

        self.Y_hat = Y_hat

        L = cross_entropy(Y, Y_hat)

        return L

    # (Y: nxc, Y_hat: nxc) -> dL_dA: nxc
    def backward(self):
        dL_dA = smce_prime(self.Y, self.Y_hat)

        assert dL_dA is not None
        assert dL_dA.ndim == 2
        assert dL_dA.shape[0] == self.Y.shape[0]
        assert dL_dA.shape[1] == self.feature_size

        return dL_dA
