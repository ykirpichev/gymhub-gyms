import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, B, C, M, N, K):
    torch.matmul(A, B, out=C)


def _case(m, n, k, seed, name):
    return {
        "_name": name,
        "A": cuda_rand(m, k, seed=seed, low=-1.0, high=1.0),
        "B": cuda_rand(k, n, seed=seed + 5, low=-1.0, high=1.0),
        "C": cuda_empty(m, n),
        "M": m,
        "N": n,
        "K": k,
    }


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([1, 2, 3, 4], (2, 2)),
        "B": cuda_tensor([5, 6, 7, 8], (2, 2)),
        "C": cuda_empty(2, 2),
        "M": 2,
        "N": 2,
        "K": 2,
    }
    shapes = [(1, 1, 1), (2, 3, 4), (17, 19, 13), (64, 64, 64)]
    random_cases = [
        _case(m, n, k, index + 1, f"{m}x{k} @ {k}x{n}") for index, (m, n, k) in enumerate(shapes)
    ]
    return [known] + random_cases


def performance_case(seed: int = 123):
    return _case(512, 512, 512, seed, "perf")


def traffic(case):
    m, n, k = case["M"], case["N"], case["K"]
    return (m * k + k * n + m * n) * 4


def flops(case):
    # One multiply and one add per (m, n, k) triple.
    return 2 * case["M"] * case["N"] * case["K"]


def build() -> Problem:
    return Problem(
        slug="matmul",
        title="Matrix multiply",
        difficulty="medium",
        summary="C = A @ B for row-major float32 matrices.",
        atol=1e-3,
        rtol=1e-3,
        params=[
            Param("A", "tensor"),
            Param("B", "tensor"),
            Param("C", "tensor", output=True),
            Param("M", "int"),
            Param("N", "int"),
            Param("K", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
        flops=flops,
    )
