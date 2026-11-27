import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(X, Y, N):
    Y.copy_(torch.nn.functional.gelu(X, approximate="tanh"))


def _case(n, seed, name):
    return {
        "_name": name,
        "X": cuda_rand(n, seed=seed, low=-6.0, high=6.0),
        "Y": cuda_empty(n),
        "N": n,
    }


def functional_cases():
    known = {
        "_name": "example",
        "X": cuda_tensor([-2.0, -1.0, 0.0, 1.0, 2.0]),
        "Y": cuda_empty(5),
        "N": 5,
    }
    sizes = [1, 3, 17, 1000, 100_003]
    random_cases = [_case(n, index + 1, f"n={n}") for index, n in enumerate(sizes)]
    return [known] + random_cases


def performance_case(seed: int = 123):
    return _case(1 << 24, seed, "perf")


def traffic(case):
    return 2 * case["N"] * 4


def build() -> Problem:
    return Problem(
        slug="gelu",
        title="GELU activation",
        difficulty="easy",
        summary="Y = GELU(X) with the tanh approximation, elementwise.",
        atol=1e-5,
        rtol=1e-4,
        params=[
            Param("X", "tensor"),
            Param("Y", "tensor", output=True),
            Param("N", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
