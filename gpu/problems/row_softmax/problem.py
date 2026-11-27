import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(X, Y, rows, cols):
    torch.softmax(X, dim=-1, out=Y)


def _case(rows, cols, seed, name):
    return {
        "_name": name,
        "X": cuda_rand(rows, cols, seed=seed, low=-4.0, high=4.0),
        "Y": cuda_empty(rows, cols),
        "rows": rows,
        "cols": cols,
    }


def functional_cases():
    known = {
        "_name": "example",
        "X": cuda_tensor([1.0, 2.0, 3.0], (1, 3)),
        "Y": cuda_empty(1, 3),
        "rows": 1,
        "cols": 3,
    }
    shapes = [(1, 1), (1, 17), (4, 8), (8, 128)]
    random_cases = [
        _case(rows, cols, index + 1, f"{rows}x{cols}") for index, (rows, cols) in enumerate(shapes)
    ]
    return [known] + random_cases


def performance_case(seed: int = 123):
    return _case(128, 4096, seed, "perf")


def traffic(case):
    return 2 * case["rows"] * case["cols"] * 4


def build() -> Problem:
    return Problem(
        slug="row_softmax",
        title="Row softmax",
        difficulty="medium",
        summary="Softmax over the last dimension of a matrix.",
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
