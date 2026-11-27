import torch

from gymkit.spec import Problem


def reference(x, weight, eps):
    rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * rms * weight


def _case(shape, seed, name, eps=1e-6):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    dim = shape[-1]
    return {
        "_name": name,
        "x": torch.randn(*shape, device="cuda", generator=generator),
        "weight": 1.0 + 0.1 * torch.randn(dim, device="cuda", generator=generator),
        "eps": eps,
    }


def functional_cases():
    example = {
        "_name": "example",
        "x": torch.tensor([[1.0, 2.0, 3.0, 4.0]], device="cuda"),
        "weight": torch.ones(4, device="cuda"),
        "eps": 0.0,
    }
    shapes = [(1, 1), (3, 7), (2, 5, 64), (4, 16, 1000), (2, 3, 4, 128)]
    cases = [example] + [_case(shape, index, f"shape={list(shape)}") for index, shape in enumerate(shapes)]
    cases.append(_case((8, 256), 99, "large eps", eps=0.5))
    return cases


def performance_case(seed: int = 0):
    return _case((16, 1024, 4096), seed, "perf")


def build() -> Problem:
    return Problem(
        slug="rms_norm",
        title="RMSNorm",
        difficulty="easy",
        summary="Root-mean-square layer norm over the last dimension, with a learned scale.",
        params=["x", "weight", "eps"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-5,
        rtol=1e-5,
    )
