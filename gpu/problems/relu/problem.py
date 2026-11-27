import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, C, N):
    torch.clamp(A, min=0, out=C)


def _case(n, seed, name, low=-2.0, high=2.0):
    return {
        "_name": name,
        "A": cuda_rand(n, seed=seed, low=low, high=high),
        "C": cuda_empty(n),
        "N": n,
    }


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([-1.0, 0.0, 2.0]),
        "C": cuda_empty(3),
        "N": 3,
    }
    sizes = [1, 17, 256, 1024]
    return [known] + [_case(n, index + 1, f"n={n}") for index, n in enumerate(sizes)]


def performance_case(seed: int = 123):
    return _case(1 << 24, seed, "perf")


def traffic(case):
    return 2 * case["N"] * 4


def build() -> Problem:
    return Problem(
        slug="relu",
        title="ReLU",
        difficulty="easy",
        summary="C[i] = max(A[i], 0).",
        atol=0,
        rtol=0,
        params=[
            Param("A", "tensor"),
            Param("C", "tensor", output=True),
            Param("N", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
