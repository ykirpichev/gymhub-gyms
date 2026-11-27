import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(X, Y, rows, cols, k):
    torch.topk(X, k, dim=-1, largest=True, sorted=True, out=(Y, torch.empty_like(Y, dtype=torch.long)))


def _case(rows, cols, k, seed, name, low=-10.0, high=10.0, rounded=False):
    X = cuda_rand(rows, cols, seed=seed, low=low, high=high)
    if rounded:
        X = X.round_().contiguous()  # many equal values
    return {
        "_name": name,
        "X": X,
        "Y": cuda_empty(rows, k),
        "rows": rows,
        "cols": cols,
        "k": k,
    }


def functional_cases():
    known = {
        "_name": "example",
        "X": cuda_tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0], (2, 5)),
        "Y": cuda_empty(2, 2),
        "rows": 2,
        "cols": 5,
        "k": 2,
    }
    shapes = [(1, 1, 1), (3, 10, 10), (4, 100, 1), (8, 1000, 7), (16, 4099, 32), (64, 32000, 50)]
    random_cases = [
        _case(r, c, k, index + 1, f"{r}x{c}, k={k}") for index, (r, c, k) in enumerate(shapes)
    ]
    ties = _case(8, 5000, 40, 21, "8x5000, k=40, many ties", rounded=True)
    negative = _case(4, 300, 20, 23, "4x300, k=20, all negative", low=-100.0, high=-50.0)
    large = _case(256, 50257, 64, 99, "256x50257, k=64")
    return [known] + random_cases + [ties, negative, large]


def performance_case(seed: int = 123):
    return _case(512, 32768, 50, seed, "perf")


def traffic(case):
    return (case["rows"] * case["cols"] + case["rows"] * case["k"]) * 4


def build() -> Problem:
    return Problem(
        slug="top_k",
        title="Top-k per row",
        difficulty="hard",
        summary="The k largest values of each row, largest first, as used when sampling from a language model.",
        atol=0.0,
        rtol=0.0,
        params=[
            Param("X", "tensor"),
            Param("Y", "tensor", output=True),
            Param("rows", "int"),
            Param("cols", "int"),
            Param("k", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
