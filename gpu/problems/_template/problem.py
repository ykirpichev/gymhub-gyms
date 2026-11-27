import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, B, C, N):
    # Replace this with the operation you want to practice.
    torch.add(A, B, out=C)


def _case(n, seed, name):
    return {
        "_name": name,
        "A": cuda_rand(n, seed=seed),
        "B": cuda_rand(n, seed=seed + 17),
        "C": cuda_empty(n),
        "N": n,
    }


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([1.0, 2.0, 3.0]),
        "B": cuda_tensor([4.0, 5.0, 6.0]),
        "C": cuda_empty(3),
        "N": 3,
    }
    return [known, _case(128, 1, "n=128")]


def performance_case(seed: int = 123):
    return _case(1 << 20, seed, "perf")


def traffic(case):
    return 3 * case["N"] * 4


def build() -> Problem:
    return Problem(
        slug="__SLUG__",
        title="__TITLE__",
        difficulty="easy",
        summary="Replace this summary. The scaffold is vector add until you change reference().",
        atol=1e-5,
        rtol=1e-5,
        params=[
            Param("A", "tensor"),
            Param("B", "tensor"),
            Param("C", "tensor", output=True),
            Param("N", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
