import math

import torch

from gymkit.spec import Problem


def reference(k_cache, v_cache, lengths, q_new, k_new, v_new):
    b, _, l, dh = k_cache.shape
    rows = torch.arange(b, device=k_cache.device)
    k = k_cache.clone()
    v = v_cache.clone()
    k[rows, :, lengths] = k_new[:, :, 0]
    v[rows, :, lengths] = v_new[:, :, 0]
    scores = (q_new @ k.transpose(-2, -1)) / math.sqrt(dh)  # [B, H, 1, L]
    future = torch.arange(l, device=k.device)[None, :] > lengths[:, None]  # [B, L]
    scores = scores.masked_fill(future[:, None, None, :], float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def _case(b, h, l, dh, lengths, seed, name, junk=100.0):
    """Cache slots at and after lengths[b] hold large junk: it must never reach the output."""
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    lengths = torch.tensor(lengths, dtype=torch.int64, device="cuda")
    stale = torch.arange(l, device="cuda")[None, None, :, None] >= lengths[:, None, None, None]
    k_cache = torch.where(stale, junk * randn(b, h, l, dh), randn(b, h, l, dh))
    v_cache = torch.where(stale, junk * randn(b, h, l, dh), randn(b, h, l, dh))
    return {
        "_name": name,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "lengths": lengths,
        "q_new": randn(b, h, 1, dh),
        "k_new": randn(b, h, 1, dh),
        "v_new": randn(b, h, 1, dh),
    }


def functional_cases():
    cuda = {"device": "cuda"}
    example = {
        "_name": "example",
        "k_cache": torch.tensor([[[[1.0, 0.0], [9.0, 9.0], [9.0, 9.0]]]], **cuda),
        "v_cache": torch.tensor([[[[1.0, 2.0], [9.0, 9.0], [9.0, 9.0]]]], **cuda),
        "lengths": torch.tensor([1], **cuda),
        "q_new": torch.tensor([[[[0.0, 2.0]]]], **cuda),
        "k_new": torch.tensor([[[[1.0, 1.0]]]], **cuda),
        "v_new": torch.tensor([[[[3.0, 4.0]]]], **cuda),
    }
    return [
        example,
        _case(1, 1, 4, 4, [0], 1, "empty cache (lengths=0)"),
        _case(1, 2, 8, 8, [7], 2, "fills the last slot"),
        _case(3, 2, 16, 8, [0, 5, 15], 3, "lengths 0, 5, 15"),
        _case(4, 8, 128, 64, [100, 3, 127, 64], 4, "varying lengths, Dh=64"),
        _case(5, 4, 300, 128, [299, 1, 150, 0, 42], 5, "Dh=128, L=300"),
        _case(2, 3, 64, 32, [10, 50], 6, "huge junk past lengths", junk=1e4),
    ]


def performance_case(seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    l = 4096
    lengths = torch.randint(l // 8, l, (8,), generator=generator).tolist()
    return _case(8, 32, l, 128, lengths, seed, "perf")


def build() -> Problem:
    return Problem(
        slug="kv_cache_decode",
        title="KV-cache decode step",
        difficulty="hard",
        summary="One decoding step of attention against a padded KV cache with a different valid length per sequence.",
        params=["k_cache", "v_cache", "lengths", "q_new", "k_new", "v_new"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-4,
        rtol=1e-4,
    )
