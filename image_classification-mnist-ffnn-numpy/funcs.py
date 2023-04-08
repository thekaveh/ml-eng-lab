import numpy as np

from consts import Consts

class Funcs:
    linear = staticmethod(lambda X, W, b: np.matmul(X, W.T) + b[:, np.newaxis].T)

    relu = staticmethod(lambda X: np.where(X > 0, X, 0))
    relu_prime = staticmethod(lambda Z: np.where(Z > 0, 1, 0))

    parametric_relu = staticmethod(lambda X, alpha: np.where(X > 0, X, alpha * X))
    parametric_relu_prime = staticmethod(lambda Z, alpha: np.where(Z > 0, 1, alpha))

    cross_entropy = staticmethod(lambda Y, Y_hat: np.mean(np.multiply(-Y, np.log(Y_hat + Consts.EPSILON)).sum(axis=1)))

    smce_prime = staticmethod(lambda Y, Y_hat: Y_hat - Y)

    @staticmethod
    # A: nxc -> Y_hat: nxc
    def softmax(A: np.matrix) -> np.matrix:
        assert A is not None
        assert A.ndim == 2

        exps = np.exp(A - A.max(axis=1)[:, np.newaxis])
        Y_hat = exps / exps.sum(axis=1)[:, np.newaxis]

        assert Y_hat is not None
        assert Y_hat.ndim == 2
        assert Y_hat.shape[0] == A.shape[0]
        assert Y_hat.shape[1] == A.shape[1]

        return Y_hat