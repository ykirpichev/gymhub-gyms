import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(X, weight, Y, rows, cols, eps):
    # float64 so the reference itself adds no rounding error.
    x = X.double()
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    Y.copy_((x * rms * weight.double()).float())


def _case(rows, cols, seed, name, eps=1e-6, low=-3.0, high=3.0):
    return {
        "_name": name,
        "X": cuda_rand(rows, cols, seed=seed, low=low, high=high),
        "weight": cuda_rand(cols, seed=seed + 3, low=0.5, high=1.5),
        "Y": cuda_empty(rows, cols),
        "rows": rows,
        "cols": cols,
        "eps": eps,
    }


def functional_cases():
    known = {
        "_name": "example",
        "X": cuda_tensor([1.0, 2.0, 3.0, 4.0], (1, 4)),
        "weight": cuda_tensor([1.0, 1.0, 2.0, 0.5]),
        "Y": cuda_empty(1, 4),
        "rows": 1,
        "cols": 4,
        "eps": 1e-6,
    }
    shapes = [(1, 1), (3, 17), (5, 1000), (64, 1024), (7, 4099)]
    random_cases = [
        _case(rows, cols, index + 1, f"{rows}x{cols}") for index, (rows, cols) in enumerate(shapes)
    ]
    # A row of zeros: the answer is 0, and only eps keeps rsqrt finite.
    zeros = _case(2, 64, 11, "2x64, all zeros", low=0.0, high=0.0)
    big_eps = _case(4, 33, 13, "4x33, eps=0.5", eps=0.5, low=-0.5, high=0.5)
    large = _case(2048, 4096, 99, "2048x4096")
    return [known] + random_cases + [zeros, big_eps, large]


def performance_case(seed: int = 123):
    return _case(8192, 4096, seed, "perf")


def traffic(case):
    rows, cols = case["rows"], case["cols"]
    return (2 * rows * cols + cols) * 4


def build() -> Problem:
    return Problem(
        slug="rms_norm",
        title="RMSNorm",
        difficulty="medium",
        summary="Divide each row by its root mean square, then scale by a weight.",
        atol=1e-4,
        rtol=1e-4,
        params=[
            Param("X", "tensor"),
            Param("weight", "tensor"),
            Param("Y", "tensor", output=True),
            Param("rows", "int"),
            Param("cols", "int"),
            Param("eps", "float"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
