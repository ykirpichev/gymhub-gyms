import math

import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(X, cos, sin, Y, T, H, D):
    half = D // 2
    x = X.view(T, H, D)
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos.view(T, 1, half), sin.view(T, 1, half)
    y = Y.view(T, H, D)
    y[..., :half] = x1 * c - x2 * s
    y[..., half:] = x2 * c + x1 * s


def _tables(T, D, base=10000.0):
    # The angle tables a model computes once and caches: angle[t, i] = t / base^(2i / D).
    half = D // 2
    inv_freq = 1.0 / (base ** (torch.arange(half, dtype=torch.float64) * 2.0 / D))
    angle = torch.arange(T, dtype=torch.float64)[:, None] * inv_freq[None, :]
    return angle.cos().float().cuda().contiguous(), angle.sin().float().cuda().contiguous()


def _case(T, H, D, seed, name):
    cos, sin = _tables(T, D)
    return {
        "_name": name,
        "X": cuda_rand(T, H, D, seed=seed, low=-2.0, high=2.0),
        "cos": cos,
        "sin": sin,
        "Y": cuda_empty(T, H, D),
        "T": T,
        "H": H,
        "D": D,
    }


def functional_cases():
    c, s = math.cos(1.0), math.sin(1.0)
    known = {
        "_name": "example",
        "X": cuda_tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], (2, 1, 4)),
        "cos": cuda_tensor([1.0, 1.0, c, 1.0], (2, 2)),
        "sin": cuda_tensor([0.0, 0.0, s, 0.0], (2, 2)),
        "Y": cuda_empty(2, 1, 4),
        "T": 2,
        "H": 1,
        "D": 4,
    }
    shapes = [(1, 1, 2), (3, 2, 8), (17, 4, 64), (100, 8, 128), (33, 5, 6)]
    random_cases = [
        _case(t, h, d, index + 1, f"T={t},H={h},D={d}") for index, (t, h, d) in enumerate(shapes)
    ]
    large = _case(2048, 32, 128, 99, "T=2048,H=32,D=128")
    return [known] + random_cases + [large]


def performance_case(seed: int = 123):
    return _case(8192, 32, 128, seed, "perf")


def traffic(case):
    T, H, D = case["T"], case["H"], case["D"]
    return (2 * T * H * D + T * D) * 4


def build() -> Problem:
    return Problem(
        slug="rope",
        title="Rotary embeddings (RoPE)",
        difficulty="medium",
        summary="Rotate pairs of query features by an angle that depends on the token position.",
        atol=1e-5,
        rtol=1e-5,
        params=[
            Param("X", "tensor"),
            Param("cos", "tensor"),
            Param("sin", "tensor"),
            Param("Y", "tensor", output=True),
            Param("T", "int"),
            Param("H", "int"),
            Param("D", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
