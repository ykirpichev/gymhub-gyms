import numpy as np


def _log_sigmoid(z):
    # log σ(z) = min(z, 0) - log(1 + exp(-|z|)); the exp never overflows.
    return np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))


def solve(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    arrays = [np.asarray(v, dtype=np.float64) for v in (policy_chosen, policy_rejected, ref_chosen, ref_rejected)]
    pc, pr, rc, rr = arrays
    if pc.ndim != 1 or pc.size == 0 or any(a.shape != pc.shape for a in arrays) or not beta > 0:
        raise ValueError("expected four equal-length, non-empty 1-D arrays and beta > 0")
    chosen_rewards = beta * (pc - rc)
    rejected_rewards = beta * (pr - rr)
    loss = float(-np.mean(_log_sigmoid(chosen_rewards - rejected_rewards)))
    accuracy = float(np.mean(chosen_rewards > rejected_rewards))
    return loss, chosen_rewards, rejected_rewards, accuracy
