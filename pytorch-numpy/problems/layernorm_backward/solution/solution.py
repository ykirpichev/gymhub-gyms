import numpy as np


def solve(x, gamma, beta, dy, eps):
    x = np.asarray(x, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    dy = np.asarray(dy, dtype=np.float64)
    mu = x.mean(axis=1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
    inv_std = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * inv_std
    y = gamma * x_hat + beta
    g = dy * gamma
    dx = inv_std * (g - g.mean(axis=1, keepdims=True) - x_hat * (g * x_hat).mean(axis=1, keepdims=True))
    return (y, dx, (dy * x_hat).sum(axis=0), dy.sum(axis=0))
