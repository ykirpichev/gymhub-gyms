"""Grader for contrastive_loss. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def _normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _ce_diagonal(logits):
    m = logits.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    return float(np.mean(lse - np.diag(logits)))


def _logits(img, txt, temperature):
    img = np.asarray(img, dtype=np.float64)
    txt = np.asarray(txt, dtype=np.float64)
    if img.ndim != 2 or img.shape != txt.shape or not temperature > 0:
        return None
    return _normalize(img) @ _normalize(txt).T / temperature


def reference(img, txt, temperature):
    logits = _logits(img, txt, temperature)
    if logits is None:
        return "ValueError"
    return (_ce_diagonal(logits) + _ce_diagonal(logits.T)) / 2


def call(module, args):
    """Bad shapes or temperature must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if got != want:
            raise AssertionError("expected ValueError: mismatched shapes or temperature <= 0" if want == "ValueError"
                                 else "raised ValueError, but the inputs are valid")
        return
    if isinstance(got, np.ndarray) and got.size == 1:
        got = got.item()
    if isinstance(got, (float, int, np.floating)) and not np.isfinite(got):
        raise AssertionError(f"loss: got {got}, expected {want:.6g}. Subtract the row max before exp.")
    try:
        default_compare(got, want, 1e-9, 1e-7, "loss")
    except AssertionError as exc:
        logits = _logits(case["img"], case["txt"], case["temperature"])
        one_way = {"image -> text only": _ce_diagonal(logits), "text -> image only": _ce_diagonal(logits.T)}
        for name, value in one_way.items():
            if isinstance(got, (float, int, np.floating)) and abs(got - value) <= 1e-7 * max(1.0, abs(value)):
                raise AssertionError(f"{exc}\n  that is the {name} loss; average both directions") from None
        raise


def _random(seed, n, d, temperature, name, aligned=0.0):
    rng = np.random.default_rng(seed)
    img = rng.standard_normal((n, d)) * rng.uniform(0.5, 3.0, (n, 1))
    txt = aligned * img + rng.standard_normal((n, d))
    return {"_name": name, "img": img.tolist(), "txt": txt.tolist(), "temperature": temperature}


def functional_cases():
    return [
        {"_name": "example", "img": [[1, 0], [0, 1]], "txt": [[2, 0], [1, 1]], "temperature": 1.0},
        {"_name": "single pair", "img": [[3.0, 4.0]], "txt": [[-1.0, 2.0]], "temperature": 0.07},
        {"_name": "scale does not matter", "img": [[100.0, 0.0], [0.0, 0.001]], "txt": [[2.0, 1.0], [0.5, 3.0]],
         "temperature": 0.5},
        {"_name": "zero vector", "img": [[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]], "txt": [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
         "temperature": 0.1},
        {"_name": "tiny temperature", "img": [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]],
         "txt": [[0.8, 0.6], [1.0, 0.0], [0.0, 1.0]], "temperature": 1e-4},
        _random(1, 4, 3, 0.07, "random 4x3, temperature 0.07"),
        _random(2, 6, 8, 2.0, "random 6x8, temperature 2"),
        _random(3, 128, 64, 0.05, "random 128x64, mostly aligned", aligned=2.0),
        _random(4, 300, 32, 0.01, "random 300x32, temperature 0.01", aligned=0.5),
        {"_name": "shape mismatch", "img": [[1.0, 0.0], [0.0, 1.0]], "txt": [[1.0, 0.0]], "temperature": 1.0},
        {"_name": "zero temperature", "img": [[1.0, 0.0]], "txt": [[1.0, 0.0]], "temperature": 0.0},
    ]


def build() -> Problem:
    return Problem(
        slug="contrastive_loss",
        title="Symmetric contrastive loss",
        difficulty="medium",
        summary="CLIP-style InfoNCE over a batch of paired embeddings: normalize, scale by temperature, cross-entropy both ways.",
        params=["img", "txt", "temperature"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "embeddings", "InfoNCE", "numerical stability"],
    )
