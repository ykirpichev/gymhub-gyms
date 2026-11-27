import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, B, N):
    B.copy_(torch.flip(A, dims=[0]))


def _case(n, seed, name):
    return {
        "_name": name,
        "A": cuda_rand(n, seed=seed, low=-1.0, high=1.0),
        "B": cuda_empty(n),
        "N": n,
    }


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        "B": cuda_empty(5),
        "N": 5,
    }
    sizes = [1, 2, 17, 256, 1000, 100_003]
    random_cases = [_case(n, index + 1, f"n={n}") for index, n in enumerate(sizes)]
    return [known] + random_cases


def performance_case(seed: int = 123):
    return _case(1 << 24, seed, "perf")


def traffic(case):
    return 2 * case["N"] * 4


def build() -> Problem:
    return Problem(
        slug="reverse",
        title="Reverse a vector",
        difficulty="easy",
        summary="B[i] = A[N - 1 - i].",
        atol=0.0,
        rtol=0.0,
        params=[
            Param("A", "tensor"),
            Param("B", "tensor", output=True),
            Param("N", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
