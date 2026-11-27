import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(ids, table, out, N, V, D):
    torch.index_select(table, 0, ids.long(), out=out)


def _ids(n, vocab, seed):
    # Whole numbers in [0, vocab), stored as float32.
    values = cuda_rand(n, seed=seed, low=0.0, high=float(vocab)).floor_()
    return values.clamp_(0.0, float(vocab - 1)).contiguous()


def _case(n, vocab, dim, seed, name):
    return {
        "_name": name,
        "ids": _ids(n, vocab, seed),
        "table": cuda_rand(vocab, dim, seed=seed + 5, low=-1.0, high=1.0),
        "out": cuda_empty(n, dim),
        "N": n,
        "V": vocab,
        "D": dim,
    }


def functional_cases():
    known = {
        "_name": "example",
        "ids": cuda_tensor([2.0, 0.0, 2.0]),
        "table": cuda_tensor([0.0, 1.0, 10.0, 11.0, 20.0, 21.0], (3, 2)),
        "out": cuda_empty(3, 2),
        "N": 3,
        "V": 3,
        "D": 2,
    }
    shapes = [(1, 1, 1), (5, 3, 7), (100, 50, 64), (1000, 32000, 128), (333, 1000, 1027)]
    random_cases = [
        _case(n, v, d, index + 1, f"N={n},V={v},D={d}") for index, (n, v, d) in enumerate(shapes)
    ]
    large = _case(8192, 50000, 768, 99, "N=8192,V=50000,D=768")
    return [known] + random_cases + [large]


def performance_case(seed: int = 123):
    return _case(16384, 32000, 1024, seed, "perf")


def traffic(case):
    # Read each id and its table row, write the output row.
    return (case["N"] + 2 * case["N"] * case["D"]) * 4


def build() -> Problem:
    return Problem(
        slug="embedding",
        title="Embedding lookup",
        difficulty="easy",
        summary="Gather one row of a table for each token id.",
        atol=0.0,
        rtol=0.0,
        params=[
            Param("ids", "tensor"),
            Param("table", "tensor"),
            Param("out", "tensor", output=True),
            Param("N", "int"),
            Param("V", "int"),
            Param("D", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
