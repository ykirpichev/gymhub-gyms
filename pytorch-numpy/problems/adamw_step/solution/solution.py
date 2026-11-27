import numpy as np


def solve(param, grad, m, v, t, lr, beta1, beta2, eps, weight_decay):
    p = np.asarray(param, dtype=np.float64)
    g = np.asarray(grad, dtype=np.float64)
    m = beta1 * np.asarray(m, dtype=np.float64) + (1 - beta1) * g
    v = beta2 * np.asarray(v, dtype=np.float64) + (1 - beta2) * g * g
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    p = p - lr * weight_decay * p
    p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
    return (p, m, v)
