"""Grader for lora_linear. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def reference(x, W, b, A, B, alpha):
    x, W, b, A, B = (np.asarray(v, dtype=np.float64) for v in (x, W, b, A, B))
    if x.ndim != 2 or W.ndim != 2 or A.ndim != 2 or B.ndim != 2 or b.ndim != 1:
        return "ValueError"
    out_f, in_f = W.shape
    r = A.shape[0]
    if r < 1 or x.shape[1] != in_f or b.shape[0] != out_f or A.shape[1] != in_f or B.shape != (out_f, r):
        return "ValueError"
    scale = alpha / r
    y = x @ W.T + b + scale * ((x @ A.T) @ B.T)
    return y, W + scale * (B @ A), r * (in_f + out_f), out_f * in_f


def call(module, args):
    """Bad shapes must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if got != want:
            raise AssertionError("expected ValueError: the shapes are inconsistent" if want == "ValueError"
                                 else "raised ValueError, but the shapes are consistent")
        return
    if not isinstance(got, tuple) or len(got) != 4:
        raise AssertionError(f"return a tuple (y, W_merged, lora_params, full_params), got {type(got).__name__}")
    y, merged, lora, full = got
    default_compare(lora, want[2], 0, 0, "lora_params")
    default_compare(full, want[3], 0, 0, "full_params")
    try:
        default_compare(y, want[0], 1e-9, 1e-9, "y")
        default_compare(merged, want[1], 1e-9, 1e-9, "W_merged")
    except AssertionError as exc:
        r = len(case["A"])
        if r > 1 and case["alpha"] != r:
            raise AssertionError(f"{exc}\n  (the low-rank update is scaled by alpha / r = {case['alpha']} / {r})") from None
        raise
    # The merged weight must reproduce y as a plain linear layer.
    merged_y = np.asarray(case["x"], dtype=np.float64) @ np.asarray(merged, dtype=np.float64).T + np.asarray(case["b"])
    default_compare(merged_y, want[0], 1e-9, 1e-9, "x @ W_merged.T + b")


def _random(seed, n, in_f, out_f, r, alpha, name, zero_b=False):
    rng = np.random.default_rng(seed)
    B = np.zeros((out_f, r)) if zero_b else rng.standard_normal((out_f, r))
    return {
        "_name": name,
        "x": rng.standard_normal((n, in_f)).tolist(),
        "W": rng.standard_normal((out_f, in_f)).tolist(),
        "b": rng.standard_normal(out_f).tolist(),
        "A": (rng.standard_normal((r, in_f)) * 0.1).tolist(),
        "B": B.tolist(),
        "alpha": alpha,
    }


def functional_cases():
    return [
        {"_name": "example", "x": [[1, 2]], "W": [[1, 0], [0, 1], [1, 1]], "b": [0, 0, 1],
         "A": [[1, 0], [0, 1]], "B": [[1, 0], [0, 1], [0, 0]], "alpha": 4.0},
        _random(1, 3, 5, 4, 2, 16.0, "B = 0 at init: base output", zero_b=True),
        {"_name": "rank 1, alpha = 1", "x": [[1.0, -1.0, 2.0]], "W": [[0.5, 0.0, 1.0]], "b": [0.25],
         "A": [[1.0, 1.0, 1.0]], "B": [[3.0]], "alpha": 1.0},
        _random(2, 4, 6, 3, 3, 6.0, "non-square, r = 3, alpha = 6"),
        _random(3, 2, 3, 7, 4, 1.0, "alpha < r"),
        _random(4, 64, 128, 96, 8, 16.0, "random 64x128 -> 96, r = 8"),
        _random(5, 200, 256, 256, 16, 32.0, "random 200x256 -> 256, r = 16"),
        {"_name": "A has wrong in", "x": [[1.0, 2.0]], "W": [[1.0, 0.0]], "b": [0.0],
         "A": [[1.0, 0.0, 0.0]], "B": [[1.0]], "alpha": 1.0},
        {"_name": "B transposed", "x": [[1.0, 2.0]], "W": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], "b": [0.0, 0.0, 0.0],
         "A": [[1.0, 0.0], [0.0, 1.0]], "B": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "alpha": 2.0},
        {"_name": "bias has wrong length", "x": [[1.0, 2.0]], "W": [[1.0, 0.0]], "b": [0.0, 1.0],
         "A": [[1.0, 0.0]], "B": [[1.0]], "alpha": 1.0},
    ]


def build() -> Problem:
    return Problem(
        slug="lora_linear",
        title="LoRA linear layer",
        difficulty="medium",
        summary="Forward pass of a low-rank adapter on a linear layer, the merged weight, and the parameter savings.",
        params=["x", "W", "b", "A", "B", "alpha"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "fine-tuning", "low rank", "linear algebra"],
    )
