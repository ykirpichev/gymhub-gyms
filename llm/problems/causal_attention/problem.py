import torch
import torch.nn.functional as F

from gymkit.spec import Problem


def reference(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def _case(shape, seed, name, scale=1.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn():
        return torch.randn(*shape, device="cuda", generator=generator)

    return {"_name": name, "q": scale * randn(), "k": scale * randn(), "v": randn()}


def functional_cases():
    cuda = {"device": "cuda"}
    example = {
        "_name": "example",
        "q": torch.tensor([[[[0.0, 0.0], [0.0, 2.0]]]], **cuda),
        "k": torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]], **cuda),
        "v": torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], **cuda),
    }
    return [
        example,
        _case((1, 1, 1, 4), 1, "T=1"),
        _case((1, 2, 5, 8), 2, "small"),
        _case((2, 4, 64, 32), 3, "Dh=32"),
        _case((2, 3, 129, 64), 4, "odd T=129"),
        _case((1, 2, 256, 128), 5, "Dh=128"),
        _case((1, 2, 100, 64), 6, "large scores (stability)", scale=6.0),
    ]


def performance_case(seed: int = 0):
    return _case((4, 16, 1024, 64), seed, "perf")


def build() -> Problem:
    return Problem(
        slug="causal_attention",
        title="Causal self-attention",
        difficulty="medium",
        summary="Scaled dot-product attention with a causal mask, from matmuls and a stable softmax.",
        params=["q", "k", "v"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-4,
        rtol=1e-4,
    )
