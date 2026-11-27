import math

import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(Q, K, V, O, N, M, d):
    # Plain float32 matmuls. TF32 would add errors of about 1e-4 here.
    saved = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        scores = torch.matmul(Q, K.T) * (1.0 / math.sqrt(d))
        torch.matmul(torch.softmax(scores, dim=-1), V, out=O)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved


def _case(n, m, d, seed, name):
    return {
        "_name": name,
        # Range +-2 for Q and K gives scores of a few units, so the softmax is not flat.
        "Q": cuda_rand(n, d, seed=seed, low=-2.0, high=2.0),
        "K": cuda_rand(m, d, seed=seed + 5, low=-2.0, high=2.0),
        "V": cuda_rand(m, d, seed=seed + 11, low=-1.0, high=1.0),
        "O": cuda_empty(n, d),
        "N": n,
        "M": m,
        "d": d,
    }


def functional_cases():
    known = {
        "_name": "example",
        "Q": cuda_tensor([1.0, 0.0, 0.0, 2.0], (2, 2)),
        "K": cuda_tensor([1.0, 0.0, 0.0, 1.0], (2, 2)),
        "V": cuda_tensor([1.0, 2.0, 3.0, 4.0], (2, 2)),
        "O": cuda_empty(2, 2),
        "N": 2,
        "M": 2,
        "d": 2,
    }
    shapes = [(1, 1, 1), (3, 5, 4), (17, 33, 7), (100, 77, 64), (64, 300, 128), (130, 1000, 32)]
    random_cases = [
        _case(n, m, d, index + 1, f"N={n},M={m},d={d}") for index, (n, m, d) in enumerate(shapes)
    ]
    large = _case(1024, 1024, 64, 99, "N=1024,M=1024,d=64")
    return [known] + random_cases + [large]


def performance_case(seed: int = 123):
    return _case(4096, 4096, 64, seed, "perf")


def traffic(case):
    n, m, d = case["N"], case["M"], case["d"]
    return (2 * n * d + 2 * m * d) * 4


def flops(case):
    # Q @ K^T and P @ V, 2 * N * M * d each. The softmax is small next to them.
    return 4 * case["N"] * case["M"] * case["d"]


def build() -> Problem:
    return Problem(
        slug="attention",
        title="Attention",
        difficulty="hard",
        summary="Single-head scaled dot-product attention, softmax(Q K^T / sqrt(d)) V.",
        atol=1e-4,
        rtol=1e-4,
        params=[
            Param("Q", "tensor"),
            Param("K", "tensor"),
            Param("V", "tensor"),
            Param("O", "tensor", output=True),
            Param("N", "int"),
            Param("M", "int"),
            Param("d", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
        flops=flops,
    )
