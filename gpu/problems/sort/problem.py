import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, B, N):
    torch.sort(A, out=(B, torch.empty_like(B, dtype=torch.long)))


def _case(n, seed, name, low=-1000.0, high=1000.0, rounded=False):
    A = cuda_rand(n, seed=seed, low=low, high=high)
    if rounded:
        A = A.round_().contiguous()
    return {"_name": name, "A": A, "B": cuda_empty(n), "N": n}


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([3.0, -1.0, 2.5, 0.0, -7.0, 2.5]),
        "B": cuda_empty(6),
        "N": 6,
    }
    sizes = [1, 2, 31, 1000, 4097, 100_003]
    random_cases = [_case(n, index + 1, f"n={n}") for index, n in enumerate(sizes)]
    ties = _case(65_536, 21, "n=65536, values -5..5", low=-5.0, high=5.0, rounded=True)
    descending = {
        "_name": "n=20000, descending",
        "A": torch.arange(20_000, 0, -1, device="cuda", dtype=torch.float32) * 0.5,
        "B": cuda_empty(20_000),
        "N": 20_000,
    }
    large = _case(1 << 22, 99, "n=4194304")
    return [known] + random_cases + [ties, descending, large]


def performance_case(seed: int = 123):
    return _case(1 << 24, seed, "perf")


def traffic(case):
    # One read and one write: the floor no sort can beat.
    return 2 * case["N"] * 4


def build() -> Problem:
    return Problem(
        slug="sort",
        title="Sort",
        difficulty="hard",
        summary="Sort a large array of floats in ascending order.",
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
