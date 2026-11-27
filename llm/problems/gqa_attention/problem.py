import math

import torch

from gymkit.spec import Problem


def reference(q, k, v):
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1)
    v = v.repeat_interleave(group, dim=1)
    length = q.shape[-2]
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    future = torch.ones(length, length, dtype=torch.bool, device=q.device).triu(1)
    return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1) @ v


def _case(b, hq, hkv, l, dh, seed, name):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    return {"_name": name, "q": randn(b, hq, l, dh), "k": randn(b, hkv, l, dh), "v": randn(b, hkv, l, dh)}


def functional_cases():
    cuda = {"device": "cuda"}
    example = {
        "_name": "example",
        "q": torch.tensor([[[[1.0], [1.0]], [[2.0], [0.0]]]], **cuda),
        "k": torch.tensor([[[[0.0], [1.0]]]], **cuda),
        "v": torch.tensor([[[[10.0], [20.0]]]], **cuda),
    }
    return [
        example,
        _case(1, 4, 4, 8, 16, 1, "MHA: Hkv = Hq"),
        _case(2, 8, 1, 8, 16, 2, "MQA: Hkv = 1"),
        _case(2, 8, 4, 1, 32, 3, "L = 1"),
        _case(2, 8, 2, 13, 32, 4, "G = 4, odd L"),
        _case(1, 16, 2, 64, 64, 5, "G = 8"),
        _case(2, 12, 4, 100, 128, 6, "Dh = 128"),
    ]


def performance_case(seed: int = 0):
    return _case(4, 32, 8, 1024, 128, seed, "perf")


def build() -> Problem:
    return Problem(
        slug="gqa_attention",
        title="Grouped-query attention",
        difficulty="medium",
        summary="Causal attention where groups of query heads share one key/value head, without copying the KV heads.",
        params=["q", "k", "v"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-4,
        rtol=1e-4,
    )
