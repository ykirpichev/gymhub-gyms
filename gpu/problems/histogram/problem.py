import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, counts, N, bins):
    hist = torch.bincount(A.long(), minlength=bins)
    counts.copy_(hist.to(torch.float32))


def _values(n, bins, seed):
    # Whole numbers in [0, bins), stored as float32.
    values = cuda_rand(n, seed=seed, low=0.0, high=float(bins)).floor_()
    return values.clamp_(0.0, float(bins - 1)).contiguous()


def _case(n, bins, seed, name, values=None):
    return {
        "_name": name,
        "A": _values(n, bins, seed) if values is None else values,
        "counts": cuda_empty(bins),
        "N": n,
        "bins": bins,
    }


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([0.0, 1.0, 1.0, 3.0, 1.0, 0.0]),
        "counts": cuda_empty(4),
        "N": 6,
        "bins": 4,
    }
    shapes = [(1, 1), (5, 10), (1000, 7), (4097, 256), (100_003, 1024)]
    random_cases = [
        _case(n, bins, index + 1, f"n={n},bins={bins}") for index, (n, bins) in enumerate(shapes)
    ]
    # Every value lands in the same bin: the worst case for atomic contention.
    same = _case(50_000, 32, 0, "n=50000, all in bin 5", torch.full((50_000,), 5.0, device="cuda"))
    large = _case(1 << 22, 256, 99, "n=4194304,bins=256")
    return [known] + random_cases + [same, large]


def performance_case(seed: int = 123):
    return _case(1 << 24, 256, seed, "perf")


def traffic(case):
    return (case["N"] + case["bins"]) * 4


def build() -> Problem:
    return Problem(
        slug="histogram",
        title="Histogram",
        difficulty="medium",
        summary="Count how many times each whole number appears in an array.",
        atol=0.0,
        rtol=0.0,
        params=[
            Param("A", "tensor"),
            Param("counts", "tensor", output=True),
            Param("N", "int"),
            Param("bins", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
