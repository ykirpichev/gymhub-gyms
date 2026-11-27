import torch

from gymkit.spec import Problem

BASE = 10000.0


def reference(q, start):
    # Angles in float64 so the reference is the exact answer; float32 solutions pass within tolerance.
    _, _, t, dh = q.shape
    half = dh // 2
    inv_freq = BASE ** (-torch.arange(half, device=q.device, dtype=torch.float64) * 2 / dh)
    pos = torch.arange(start, start + t, device=q.device, dtype=torch.float64)
    angle = pos[:, None] * inv_freq[None, :]
    cos = angle.cos().to(q.dtype)
    sin = angle.sin().to(q.dtype)
    q1, q2 = q[..., :half], q[..., half:]
    return torch.cat([q1 * cos - q2 * sin, q2 * cos + q1 * sin], dim=-1)


def _case(shape, start, seed, name):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return {"_name": name, "q": torch.randn(*shape, device="cuda", generator=generator), "start": start}


def functional_cases():
    example = {
        "_name": "example",
        "q": torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], device="cuda"),
        "start": 1,
    }
    return [
        example,
        _case((1, 1, 1, 2), 0, 1, "position 0 is the identity"),
        _case((1, 1, 5, 2), 0, 2, "Dh=2"),
        _case((2, 3, 7, 8), 0, 3, "small"),
        _case((1, 4, 16, 64), 3000, 4, "start=3000"),
        _case((2, 8, 128, 128), 17, 5, "Dh=128, start=17"),
        _case((3, 2, 33, 96), 1000, 6, "Dh=96, odd T"),
    ]


def performance_case(seed: int = 0):
    return _case((4, 32, 2048, 128), 0, seed, "perf")


def build() -> Problem:
    return Problem(
        slug="rope",
        title="Rotary position embedding",
        difficulty="medium",
        summary="Rotate query/key channel pairs by position-dependent angles (rotate_half convention).",
        params=["q", "start"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-3,
        rtol=1e-3,
    )
