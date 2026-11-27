import torch

from gymkit.spec import Problem


def reference(x, W, b, A, B, alpha):
    r = A.shape[0]
    return torch.addmm(b, x, W.t()) + (alpha / r) * ((x @ A.t()) @ B.t())


def _case(n, d_in, d_out, r, seed, name, alpha=16.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return scale * torch.randn(*shape, device="cuda", generator=generator)

    return {
        "_name": name,
        "x": randn(n, d_in),
        "W": randn(d_out, d_in, scale=d_in**-0.5),
        "b": randn(d_out, scale=0.1),
        "A": randn(r, d_in, scale=d_in**-0.5),
        "B": randn(d_out, r, scale=0.1),
        "alpha": alpha,
    }


def functional_cases():
    cuda = {"device": "cuda"}
    example = {
        "_name": "example",
        "x": torch.tensor([[1.0, 2.0]], **cuda),
        "W": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], **cuda),
        "b": torch.tensor([0.0, 0.0, 1.0], **cuda),
        "A": torch.tensor([[1.0, 1.0]], **cuda),
        "B": torch.tensor([[1.0], [0.0], [-1.0]], **cuda),
        "alpha": 2.0,
    }
    zero_b = _case(10, 33, 17, 4, 3, "B is zero (fresh adapter)")
    zero_b["B"] = torch.zeros_like(zero_b["B"])
    return [
        example,
        _case(1, 1, 1, 1, 1, "all sizes 1"),
        _case(7, 16, 24, 2, 2, "small, r=2"),
        zero_b,
        _case(64, 512, 256, 8, 4, "rectangular, r=8", alpha=8.0),
        _case(33, 300, 700, 16, 5, "odd sizes, r=16", alpha=32.0),
        _case(5, 64, 64, 64, 6, "r equals the width", alpha=1.0),
    ]


def performance_case(seed: int = 0):
    return _case(4096, 4096, 4096, 16, seed, "perf")


def build() -> Problem:
    return Problem(
        slug="lora_linear",
        title="LoRA linear layer",
        difficulty="easy",
        summary="A frozen linear layer plus a scaled low-rank update, computed without forming the full update matrix.",
        params=["x", "W", "b", "A", "B", "alpha"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-3,
        rtol=1e-3,
    )
