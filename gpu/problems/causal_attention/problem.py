import math

import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(Q, K, V, O, H, L, d):
    # float64 so the reference adds no rounding error of its own.
    q, k, v = Q.double(), K.double(), V.double()
    scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(d))
    mask = torch.ones(L, L, dtype=torch.bool, device=Q.device).triu(1)
    scores.masked_fill_(mask, float("-inf"))
    O.copy_(torch.matmul(torch.softmax(scores, dim=-1), v).float())


def _case(h, length, d, seed, name):
    return {
        "_name": name,
        "Q": cuda_rand(h, length, d, seed=seed, low=-2.0, high=2.0),
        "K": cuda_rand(h, length, d, seed=seed + 5, low=-2.0, high=2.0),
        "V": cuda_rand(h, length, d, seed=seed + 11, low=-1.0, high=1.0),
        "O": cuda_empty(h, length, d),
        "H": h,
        "L": length,
        "d": d,
    }


def functional_cases():
    known = {
        "_name": "example",
        "Q": cuda_tensor([1.0, 0.0, 0.0, 2.0], (1, 2, 2)),
        "K": cuda_tensor([1.0, 0.0, 0.0, 1.0], (1, 2, 2)),
        "V": cuda_tensor([1.0, 2.0, 3.0, 4.0], (1, 2, 2)),
        "O": cuda_empty(1, 2, 2),
        "H": 1,
        "L": 2,
        "d": 2,
    }
    shapes = [(1, 1, 1), (1, 5, 4), (2, 17, 7), (3, 100, 64), (4, 129, 32), (2, 300, 128)]
    random_cases = [
        _case(h, length, d, index + 1, f"H={h},L={length},d={d}")
        for index, (h, length, d) in enumerate(shapes)
    ]
    large = _case(8, 1024, 64, 99, "H=8,L=1024,d=64")
    return [known] + random_cases + [large]


def performance_case(seed: int = 123):
    return _case(16, 2048, 64, seed, "perf")


def traffic(case):
    return 4 * case["H"] * case["L"] * case["d"] * 4


def flops(case):
    # Q K^T and P V over the lower triangle only: 2 * (2 * L * L * d) / 2 per head.
    return 2 * case["H"] * case["L"] * case["L"] * case["d"]


def build() -> Problem:
    return Problem(
        slug="causal_attention",
        title="Causal flash attention",
        difficulty="hard",
        summary="Multi-head causal attention without storing the score matrix, skipping the masked tiles.",
        atol=1e-4,
        rtol=1e-4,
        params=[
            Param("Q", "tensor"),
            Param("K", "tensor"),
            Param("V", "tensor"),
            Param("O", "tensor", output=True),
            Param("H", "int"),
            Param("L", "int"),
            Param("d", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
        flops=flops,
    )
