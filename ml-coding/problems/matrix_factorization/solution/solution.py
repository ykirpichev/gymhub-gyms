import numpy as np


def solve(ratings, P, Q, lr, lam, epochs):
    P = np.array(P, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same rank")
    for _ in range(epochs):
        for u, i, r in ratings:
            err = r - P[u] @ Q[i]
            p_old = P[u].copy()
            P[u] += lr * (err * Q[i] - lam * P[u])
            Q[i] += lr * (err * p_old - lam * Q[i])
    errors = [r - P[u] @ Q[i] for u, i, r in ratings]
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return P, Q, rmse
