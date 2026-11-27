import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, C, M, N):
    C.copy_(A.transpose(0, 1))


def _case(m, n, seed, name):
    return {
        "_name": name,
        "A": cuda_rand(m, n, seed=seed, low=-2.0, high=2.0),
        "C": cuda_empty(n, m),
        "M": m,
        "N": n,
    }


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([1, 2, 3, 4, 5, 6], (2, 3)),
        "C": cuda_empty(3, 2),
        "M": 2,
        "N": 3,
    }
    shapes = [(1, 1), (1, 17), (17, 1), (32, 64), (63, 65)]
    random_cases = [_case(m, n, index + 1, f"{m}x{n}") for index, (m, n) in enumerate(shapes)]
    return [known] + random_cases


def performance_case(seed: int = 123):
    return _case(4096, 4096, seed, "perf")


def traffic(case):
    return 2 * case["M"] * case["N"] * 4


def build() -> Problem:
    return Problem(
        slug="transpose",
        title="Matrix transpose",
        difficulty="easy",
        summary="C is the transpose of row-major A.",
        atol=0,
        rtol=0,
        params=[
            Param("A", "tensor"),
            Param("C", "tensor", output=True),
            Param("M", "int"),
            Param("N", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
