import torch
import torch.nn.functional as F

from gymkit.spec import Problem

IGNORE_INDEX = -100


def reference(logits, targets, label_smoothing):
    return F.cross_entropy(logits, targets, ignore_index=IGNORE_INDEX, label_smoothing=label_smoothing)


def _case(n, v, seed, name, label_smoothing=0.0, scale=1.0, ignore_frac=0.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = scale * torch.randn(n, v, device="cuda", generator=generator)
    targets = torch.randint(0, v, (n,), device="cuda", generator=generator)
    if ignore_frac > 0:
        drop = torch.rand(n, device="cuda", generator=generator) < ignore_frac
        drop[0] = False  # keep at least one real target
        targets = torch.where(drop, torch.full_like(targets, IGNORE_INDEX), targets)
    return {"_name": name, "logits": logits, "targets": targets, "label_smoothing": label_smoothing}


def functional_cases():
    example = {
        "_name": "example",
        "logits": torch.tensor([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]], device="cuda"),
        "targets": torch.tensor([0, -100], device="cuda"),
        "label_smoothing": 0.0,
    }
    one_left = _case(64, 1000, 7, "all but one ignored, large logits", scale=1000.0)
    one_left["targets"][1:] = IGNORE_INDEX
    return [
        example,
        _case(1, 2, 1, "N=1, V=2"),
        _case(16, 50, 2, "no ignored targets"),
        _case(128, 3000, 3, "some ignored", ignore_frac=0.3),
        _case(32, 500, 4, "label smoothing 0.1", label_smoothing=0.1, ignore_frac=0.2),
        _case(8, 32000, 5, "large logits + smoothing", label_smoothing=0.25, scale=200.0),
        one_left,
    ]


def performance_case(seed: int = 0):
    return _case(8192, 32000, seed, "perf", label_smoothing=0.1, ignore_frac=0.1)


def build() -> Problem:
    return Problem(
        slug="cross_entropy",
        title="Cross-entropy loss",
        difficulty="easy",
        summary="Mean token cross-entropy from logits, with ignored targets and label smoothing.",
        params=["logits", "targets", "label_smoothing"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-4,
        rtol=1e-4,
    )
