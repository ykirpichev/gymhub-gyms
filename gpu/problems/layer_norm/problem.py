import torch
import torch.nn.functional as F

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(X, gamma, beta, Y, rows, cols, eps):
    Y.copy_(F.layer_norm(X, (cols,), gamma, beta, eps))


def _case(rows, cols, seed, name, eps=1e-5):
    return {
        "_name": name,
        # An offset mean makes the one-pass E[x^2] - E[x]^2 formula lose precision.
        "X": cuda_rand(rows, cols, seed=seed, low=-1.0, high=7.0),
        "gamma": cuda_rand(cols, seed=seed + 3, low=0.5, high=1.5),
        "beta": cuda_rand(cols, seed=seed + 7, low=-0.5, high=0.5),
        "Y": cuda_empty(rows, cols),
        "rows": rows,
        "cols": cols,
        "eps": eps,
    }


def functional_cases():
    known = {
        "_name": "example",
        "X": cuda_tensor([1.0, 2.0, 3.0, 4.0], (1, 4)),
        "gamma": cuda_tensor([1.0, 1.0, 2.0, 2.0]),
        "beta": cuda_tensor([0.0, 0.0, 0.0, 1.0]),
        "Y": cuda_empty(1, 4),
        "rows": 1,
        "cols": 4,
        "eps": 1e-5,
    }
    shapes = [(1, 1), (3, 17), (5, 1000), (64, 1024), (7, 4099)]
    random_cases = [
        _case(rows, cols, index + 1, f"{rows}x{cols}") for index, (rows, cols) in enumerate(shapes)
    ]
    big_eps = _case(4, 33, 11, "4x33, eps=0.1", eps=0.1)
    large = _case(1024, 2048, 99, "1024x2048")
    return [known] + random_cases + [big_eps, large]


def performance_case(seed: int = 123):
    return _case(4096, 4096, seed, "perf")


def traffic(case):
    rows, cols = case["rows"], case["cols"]
    return (2 * rows * cols + 2 * cols) * 4


def build() -> Problem:
    return Problem(
        slug="layer_norm",
        title="Layer norm",
        difficulty="medium",
        summary="Normalize each row to mean 0 and variance 1, then scale and shift.",
        atol=1e-4,
        rtol=1e-4,
        params=[
            Param("X", "tensor"),
            Param("gamma", "tensor"),
            Param("beta", "tensor"),
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
