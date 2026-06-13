import numpy as np

from consts import Consts


def linear(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(X, W.T) + b[:, np.newaxis].T


def parametric_relu(X: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(X > 0, X, alpha * X)


def parametric_relu_prime(Z: np.ndarray, alpha: float) -> np.ndarray:
    return np.where(Z > 0, 1, alpha)


def cross_entropy(Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
    return np.mean(np.multiply(-Y, np.log(Y_hat + Consts.EPSILON)).sum(axis=1))


def smce_prime(Y: np.ndarray, Y_hat: np.ndarray) -> np.ndarray:
    return Y_hat - Y


def softmax(A: np.ndarray) -> np.ndarray:
    assert A is not None
    assert A.ndim == 2

    exps = np.exp(A - A.max(axis=1)[:, np.newaxis])
    Y_hat = exps / exps.sum(axis=1)[:, np.newaxis]

    assert Y_hat is not None
    assert Y_hat.ndim == 2
    assert Y_hat.shape[0] == A.shape[0]
    assert Y_hat.shape[1] == A.shape[1]

    return Y_hat
