import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, x, y, M, K):
    torch.mv(A, x, out=y)


def _case(m, k, seed, name):
    return {
        "_name": name,
        "A": cuda_rand(m, k, seed=seed, low=-1.0, high=1.0),
        "x": cuda_rand(k, seed=seed + 7, low=-1.0, high=1.0),
        "y": cuda_empty(m),
        "M": m,
        "K": k,
    }


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([1, 2, 3, 4, 5, 6], (2, 3)),
        "x": cuda_tensor([1, 0, -1]),
        "y": cuda_empty(2),
        "M": 2,
        "K": 3,
    }
    shapes = [(1, 1), (3, 33), (33, 1), (100, 1000), (513, 2049)]
    random_cases = [_case(m, k, index + 1, f"{m}x{k}") for index, (m, k) in enumerate(shapes)]
    return [known] + random_cases


def performance_case(seed: int = 123):
    return _case(8192, 4096, seed, "perf")


def traffic(case):
    m, k = case["M"], case["K"]
    return (m * k + k + m) * 4


def flops(case):
    return 2 * case["M"] * case["K"]


def build() -> Problem:
    return Problem(
        slug="matvec",
        title="Matrix-vector multiply",
        difficulty="medium",
        summary="y = A @ x for a row-major M x K matrix.",
        atol=1e-3,
        rtol=1e-3,
        params=[
            Param("A", "tensor"),
            Param("x", "tensor"),
            Param("y", "tensor", output=True),
            Param("M", "int"),
            Param("K", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
        flops=flops,
    )
