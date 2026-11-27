import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_randint, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(W, scale, x, y, N, K):
    y.copy_(((W.double() @ x.double()) * scale.double()).float())


def _case(n, k, seed, name):
    return {
        "_name": name,
        "W": cuda_randint(n, k, seed=seed, low=-127, high=128, dtype=torch.int8),
        "scale": cuda_rand(n, seed=seed + 3, low=0.001, high=0.02),
        "x": cuda_rand(k, seed=seed + 7, low=-1.0, high=1.0, dtype=torch.float16),
        "y": cuda_empty(n),
        "N": n,
        "K": k,
    }


def functional_cases():
    known = {
        "_name": "example",
        "W": cuda_tensor([1, -2, 3, 127, 0, -127], (2, 3), dtype=torch.int8),
        "scale": cuda_tensor([0.5, 0.01]),
        "x": cuda_tensor([1.0, 0.5, -1.0], dtype=torch.float16),
        "y": cuda_empty(2),
        "N": 2,
        "K": 3,
    }
    shapes = [(1, 1), (3, 17), (5, 1000), (64, 4096), (4096, 4096), (11008, 4096), (4096, 11008)]
    return [known] + [_case(n, k, index + 1, f"{n}x{k}") for index, (n, k) in enumerate(shapes)]


def performance_case(seed: int = 123):
    return _case(8192, 8192, seed, "perf")


def traffic(case):
    n, k = case["N"], case["K"]
    return n * k + n * 4 + k * 2 + n * 4


def build() -> Problem:
    return Problem(
        slug="int8_matvec",
        title="Int8 weight matvec",
        difficulty="medium",
        summary="y = scale * (W @ x) with int8 weights, per-row scales and an FP16 vector: quantized LLM decoding.",
        atol=1e-3,
        rtol=1e-4,
        params=[
            Param("W", "tensor", dtype="int8"),
            Param("scale", "tensor"),
            Param("x", "tensor", dtype="float16"),
            Param("y", "tensor", output=True),
            Param("N", "int"),
            Param("K", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
