import numpy as np

from funcs import Funcs
from consts import Consts

class LinearLayer:
    def __init__(
        self
        , W: np.matrix = None
        , b: np.matrix = None
        , feature_size_out: int = Consts.FEATURES_SIZE_OUT
        , feature_size_in: int = Consts.FEATURES_SIZE_IN
    ):
        assert feature_size_in is not None and feature_size_in > 0
        assert feature_size_out is not None and feature_size_out > 0

        self.feature_size_in = feature_size_in
        self.feature_size_out = feature_size_out

        if W is not None:
            self.W = np.ndarray.copy(W)
        else:
            self.W = np.random.standard_normal(size=(feature_size_out, feature_size_in))
            self.W = self.W / np.linalg.norm(self.W)

            assert self.W is not None
            assert self.W.ndim == 2
            assert self.W.shape[0] == self.feature_size_out
            assert self.W.shape[1] == self.feature_size_in
            np.testing.assert_almost_equal(np.linalg.norm(self.W), 1)

        if b is not None:
            self.b = np.ndarray.copy(b)
        else:
            self.b = np.random.standard_normal(size=feature_size_out)
            self.b = self.b / np.linalg.norm(self.b)

            assert self.b is not None
            assert self.b.ndim == 1
            assert self.b.shape[0] == self.feature_size_out
            np.testing.assert_almost_equal(np.linalg.norm(self.b), 1)

    # X: nxm -> Z: nxc
    def forward(self, X: np.matrix):
        assert X is not None
        assert X.ndim == 2
        assert X.shape[1] == self.feature_size_in

        self.X = X

        Z = Funcs.linear(self.X, self.W, self.b)

        assert Z is not None
        assert Z.ndim == 2
        assert Z.shape[0] == self.X.shape[0]
        assert Z.shape[1] == self.feature_size_out

        return Z

    # dL_dZ: nxc -> (dW: cxm, db: cx1)
    def backward(self, dL_dZ: np.matrix):
        n = self.X.shape[0]

        assert dL_dZ is not None
        assert dL_dZ.ndim == 2
        assert dL_dZ.shape[0] == n
        assert dL_dZ.shape[1] == self.feature_size_out

        dW = 1./n * dL_dZ.T @ self.X

        assert dW is not None
        assert dW.ndim == 2
        assert dW.shape[0] == self.feature_size_out
        assert dW.shape[1] == self.feature_size_in

        db = 1./n * dL_dZ.T.sum(axis=1, keepdims=True).reshape(-1)

        assert db is not None
        assert db.ndim == 1
        assert db.shape[0] == self.feature_size_out

        return dW, db
