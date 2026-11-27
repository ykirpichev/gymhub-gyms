import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, W, C, N, K):
    folded = A.unfold(0, K, 1)
    torch.mv(folded, W, out=C)


def _case(n, k, seed, name):
    return {
        "_name": name,
        "A": cuda_rand(n, seed=seed, low=-1.0, high=1.0),
        "W": cuda_rand(k, seed=seed + 9, low=-1.0, high=1.0),
        "C": cuda_empty(n - k + 1),
        "N": n,
        "K": k,
    }


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([1.0, 2.0, 3.0, 4.0]),
        "W": cuda_tensor([1.0, 0.0, -1.0]),
        "C": cuda_empty(2),
        "N": 4,
        "K": 3,
    }
    shapes = [(1, 1), (8, 1), (8, 8), (128, 5), (1000, 7)]
    random_cases = [
        _case(n, k, index + 1, f"n={n},k={k}") for index, (n, k) in enumerate(shapes)
    ]
    return [known] + random_cases


def performance_case(seed: int = 123):
    return _case(1 << 20, 7, seed, "perf")


def traffic(case):
    n = case["N"]
    k = case["K"]
    return (n + k + (n - k + 1)) * 4


def flops(case):
    # Each of the N - K + 1 outputs does K multiplies and K adds.
    return 2 * (case["N"] - case["K"] + 1) * case["K"]


def build() -> Problem:
    return Problem(
        slug="conv1d",
        title="1D convolution",
        difficulty="easy",
        summary="Valid 1D convolution of a signal with a short kernel.",
        atol=1e-4,
        rtol=1e-4,
        params=[
            Param("A", "tensor"),
            Param("W", "tensor"),
            Param("C", "tensor", output=True),
            Param("N", "int"),
            Param("K", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
        flops=flops,
    )
