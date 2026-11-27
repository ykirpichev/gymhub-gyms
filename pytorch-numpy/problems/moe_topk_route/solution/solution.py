import numpy as np


def solve(router_logits, expert_outputs, k):
    logits = np.asarray(router_logits, dtype=np.float64)
    outs = np.asarray(expert_outputs, dtype=np.float64)
    tokens, num_experts = logits.shape
    experts = np.argsort(-logits, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(logits, experts, axis=1)
    gates = np.exp(top - top.max(axis=1, keepdims=True))
    gates = gates / gates.sum(axis=1, keepdims=True)
    chosen = outs[experts, np.arange(tokens)[:, None]]
    output = (gates[:, :, None] * chosen).sum(axis=1)
    load = np.bincount(experts.ravel(), minlength=num_experts)
    return (output, experts, gates, load)
