import torch
import torch.nn.functional as F

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(X, Y, rows, cols):
    gate, up = X[:, :cols], X[:, cols:]
    torch.mul(F.silu(gate), up, out=Y)


def _case(rows, cols, seed, name, low=-6.0, high=6.0):
    return {
        "_name": name,
        "X": cuda_rand(rows, 2 * cols, seed=seed, low=low, high=high),
        "Y": cuda_empty(rows, cols),
        "rows": rows,
        "cols": cols,
    }


def functional_cases():
    known = {
        "_name": "example",
        "X": cuda_tensor([0.0, 1.0, 2.0, 3.0, -1.0, 2.0, 0.5, 4.0], (2, 4)),
        "Y": cuda_empty(2, 2),
        "rows": 2,
        "cols": 2,
    }
    shapes = [(1, 1), (3, 5), (16, 1000), (7, 4097), (256, 2048)]
    random_cases = [
        _case(rows, cols, index + 1, f"{rows}x{cols}") for index, (rows, cols) in enumerate(shapes)
    ]
    # Large gates: exp(-x) overflows to inf for x below about -88.
    extreme = _case(4, 257, 21, "4x257, gates in [-200, 200]", low=-200.0, high=200.0)
    large = _case(2048, 8192, 99, "2048x8192")
    return [known] + random_cases + [extreme, large]


def performance_case(seed: int = 123):
    return _case(4096, 11008, seed, "perf")


def traffic(case):
    return 3 * case["rows"] * case["cols"] * 4


def build() -> Problem:
    return Problem(
        slug="swiglu",
        title="Fused SwiGLU",
        difficulty="easy",
        summary="SiLU of the gate half times the up half, in one pass.",
        atol=1e-4,
        rtol=1e-4,
        params=[
            Param("X", "tensor"),
            Param("Y", "tensor", output=True),
            Param("rows", "int"),
            Param("cols", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
