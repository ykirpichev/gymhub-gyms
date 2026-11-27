import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, B, N):
    torch.cumsum(A, dim=0, out=B)


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
        "A": cuda_tensor([3.0, 1.0, 4.0, 1.0, 5.0]),
        "B": cuda_empty(5),
        "N": 5,
    }
    sizes = [1, 2, 31, 256, 1000, 4097, 100_003]
    random_cases = [_case(n, index + 1, f"n={n}") for index, n in enumerate(sizes)]
    return [known] + random_cases


def performance_case(seed: int = 123):
    return _case(1 << 22, seed, "perf")


def traffic(case):
    return 2 * case["N"] * 4


def build() -> Problem:
    return Problem(
        slug="prefix_sum",
        title="Prefix sum",
        difficulty="medium",
        summary="Inclusive scan: B[i] = A[0] + A[1] + ... + A[i].",
        # Rounding error in a float32 scan grows with N. With inputs in [-1, 1] and N = 4M, a
        # block-wise parallel scan is off by about 1e-3 and a single-thread loop by about 4e-2.
        # Real bugs (exclusive scan, a missing block offset, a skipped element) are off by ~0.5 or more.
        atol=1e-1,
        rtol=1e-5,
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
