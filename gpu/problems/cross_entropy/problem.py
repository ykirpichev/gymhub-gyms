import torch
import torch.nn.functional as F

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(logits, targets, loss, rows, V):
    loss.copy_(F.cross_entropy(logits.double(), targets.long(), reduction="none").float())


def _targets(rows, vocab, seed):
    values = cuda_rand(rows, seed=seed, low=0.0, high=float(vocab)).floor_()
    return values.clamp_(0.0, float(vocab - 1)).contiguous()


def _case(rows, vocab, seed, name, low=-8.0, high=8.0):
    return {
        "_name": name,
        "logits": cuda_rand(rows, vocab, seed=seed, low=low, high=high),
        "targets": _targets(rows, vocab, seed + 7),
        "loss": cuda_empty(rows),
        "rows": rows,
        "V": vocab,
    }


def functional_cases():
    known = {
        "_name": "example",
        "logits": cuda_tensor([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], (2, 3)),
        "targets": cuda_tensor([2.0, 0.0]),
        "loss": cuda_empty(2),
        "rows": 2,
        "V": 3,
    }
    shapes = [(1, 1), (3, 10), (8, 1000), (17, 4099), (64, 32000)]
    random_cases = [
        _case(rows, vocab, index + 1, f"{rows}x{vocab}") for index, (rows, vocab) in enumerate(shapes)
    ]
    # exp(1000) is inf in float32: only a max-subtracted log-sum-exp survives.
    huge = _case(4, 5000, 21, "4x5000, logits near 1000", low=990.0, high=1010.0)
    negative = _case(4, 777, 23, "4x777, logits near -1000", low=-1010.0, high=-990.0)
    large = _case(512, 50257, 99, "512x50257")
    return [known] + random_cases + [huge, negative, large]


def performance_case(seed: int = 123):
    return _case(1024, 32768, seed, "perf")


def traffic(case):
    rows, vocab = case["rows"], case["V"]
    return (rows * vocab + 2 * rows) * 4


def build() -> Problem:
    return Problem(
        slug="cross_entropy",
        title="Cross-entropy",
        difficulty="medium",
        summary="Per-row loss of a softmax over a large vocabulary, in one pass with an online log-sum-exp.",
        atol=1e-4,
        rtol=1e-4,
        params=[
            Param("logits", "tensor"),
            Param("targets", "tensor"),
            Param("loss", "tensor", output=True),
            Param("rows", "int"),
            Param("V", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
    )
