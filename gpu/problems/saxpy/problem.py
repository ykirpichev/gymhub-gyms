import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, B, C, alpha, N):
    torch.add(B, A, alpha=alpha, out=C)


def _case(n, seed, alpha, name):
    return {
        "_name": name,
        "A": cuda_rand(n, seed=seed, low=-1.0, high=1.0),
        "B": cuda_rand(n, seed=seed + 3, low=-1.0, high=1.0),
        "C": cuda_empty(n),
        "alpha": float(alpha),
        "N": n,
    }


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([1.0, 2.0]),
        "B": cuda_tensor([3.0, 4.0]),
        "C": cuda_empty(2),
        "alpha": 2.0,
        "N": 2,
    }
    sizes = [(1, 0.0), (17, -1.5), (256, 0.5), (1000, 3.25)]
    random_cases = [
        _case(n, index + 1, alpha, f"n={n}") for index, (n, alpha) in enumerate(sizes)
    ]
    return [known] + random_cases


def performance_case(seed: int = 123):
    return _case(1 << 24, seed, 1.25, "perf")


def traffic(case):
    return 3 * case["N"] * 4


def build() -> Problem:
    return Problem(
        slug="saxpy",
        title="SAXPY",
        difficulty="easy",
        summary="C = alpha * A + B.",
        atol=1e-5,
        rtol=1e-5,
        params=[
            Param("A", "tensor"),
            Param("B", "tensor"),
            Param("C", "tensor", output=True),
            Param("alpha", "float"),
            Param("N", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
