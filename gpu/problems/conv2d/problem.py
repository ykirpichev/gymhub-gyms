import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(A, F, C, H, W, KH, KW):
    # cuDNN may use TF32 by default. Turn it off so the reference is plain float32.
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=False):
        out = torch.nn.functional.conv2d(A.view(1, 1, H, W), F.view(1, 1, KH, KW))
    C.copy_(out.view(H - KH + 1, W - KW + 1))


def _case(h, w, kh, kw, seed, name):
    return {
        "_name": name,
        "A": cuda_rand(h, w, seed=seed, low=-1.0, high=1.0),
        "F": cuda_rand(kh, kw, seed=seed + 9, low=-1.0, high=1.0),
        "C": cuda_empty(h - kh + 1, w - kw + 1),
        "H": h,
        "W": w,
        "KH": kh,
        "KW": kw,
    }


def functional_cases():
    known = {
        "_name": "example",
        "A": cuda_tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], (3, 3)),
        "F": cuda_tensor([1, 2, 3, 4], (2, 2)),
        "C": cuda_empty(2, 2),
        "H": 3,
        "W": 3,
        "KH": 2,
        "KW": 2,
    }
    shapes = [
        (1, 1, 1, 1),
        (5, 4, 5, 4),  # kernel as large as the image: one output
        (33, 65, 3, 5),
        (100, 37, 7, 1),
        (19, 300, 1, 9),
        (257, 129, 5, 5),
        (40, 40, 15, 15),
    ]
    random_cases = [
        _case(h, w, kh, kw, index + 1, f"{h}x{w} * {kh}x{kw}")
        for index, (h, w, kh, kw) in enumerate(shapes)
    ]
    large = _case(1000, 1000, 7, 7, 99, "1000x1000 * 7x7")
    return [known] + random_cases + [large]


def performance_case(seed: int = 123):
    return _case(4096, 4096, 7, 7, seed, "perf")


def _out(case):
    return (case["H"] - case["KH"] + 1) * (case["W"] - case["KW"] + 1)


def traffic(case):
    return (case["H"] * case["W"] + case["KH"] * case["KW"] + _out(case)) * 4


def flops(case):
    return 2 * _out(case) * case["KH"] * case["KW"]


def build() -> Problem:
    return Problem(
        slug="conv2d",
        title="2D convolution",
        difficulty="medium",
        summary="Valid 2D convolution of an image with a small filter.",
        atol=1e-4,
        rtol=1e-4,
        params=[
            Param("A", "tensor"),
            Param("F", "tensor"),
            Param("C", "tensor", output=True),
            Param("H", "int"),
            Param("W", "int"),
            Param("KH", "int"),
            Param("KW", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
        flops=flops,
    )
