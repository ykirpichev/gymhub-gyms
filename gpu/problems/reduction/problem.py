import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, out, N):
    out.copy_(A.sum().reshape(1))


def _case(n, seed, name, low=0.0, high=1.0):
    return {
        "_name": name,
        "A": cuda_rand(n, seed=seed, low=low, high=high),
        "out": cuda_empty(1),
        "N": n,
    }


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([1.0, 2.0, 3.0, 4.0]),
        "out": cuda_empty(1),
        "N": 4,
    }
    sizes = [1, 17, 256, 1024, 100_003]
    random_cases = [_case(n, index + 1, f"n={n}", low=-1.0, high=1.0) for index, n in enumerate(sizes)]
    return [known] + random_cases


def performance_case(seed: int = 123):
    return _case(1 << 24, seed, "perf")


def traffic(case):
    return (case["N"] + 1) * 4


def build() -> Problem:
    return Problem(
        slug="reduction",
        title="Sum reduction",
        difficulty="medium",
        summary="out[0] is the sum of A.",
        atol=1e-1,
        rtol=1e-2,
        params=[
            Param("A", "tensor"),
            Param("out", "tensor", output=True),
            Param("N", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
