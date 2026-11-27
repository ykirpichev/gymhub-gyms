"""Grader for causal_attention_mask. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def reference(valid):
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2:
        valid = valid.reshape(len(valid), 0)  # empty batch: shape (0, 0)
    length = valid.shape[1]
    causal = np.tril(np.ones((length, length), dtype=bool))
    return causal[None] & valid[:, :, None] & valid[:, None, :]


def compare(got, want, case):
    g = np.asarray(got)
    if g.shape != want.shape:
        raise AssertionError(f"mask: shape {list(g.shape)} != expected {list(want.shape)} (B, L, L)")
    if g.dtype != np.bool_:
        raise AssertionError(f"mask: dtype {g.dtype}, expected bool")
    bad = np.argwhere(g != want)
    if len(bad):
        b, i, j = (int(x) for x in bad[0])
        why = ("j > i is in the future" if j > i else
               f"query {i} is padding" if not case["valid"][b][i] else
               f"key {j} is padding" if not case["valid"][b][j] else
               "j <= i and both tokens are real")
        raise AssertionError(f"mask: {len(bad)} of {want.size} entries wrong; first at [b={b}, i={i}, j={j}]: "
                             f"got {bool(g[b, i, j])}, expected {bool(want[b, i, j])} ({why})")


def functional_cases():
    rng = np.random.default_rng(7)
    lengths = rng.integers(0, 33, 16)
    right = [[j < n for j in range(32)] for n in lengths]
    left = [[j >= 32 - n for j in range(32)] for n in lengths]
    return [
        {"_name": "example", "valid": [[True, True, True], [True, True, False]]},
        {"_name": "no padding", "valid": [[True] * 5]},
        {"_name": "left padding (keys must be checked)", "valid": [[False, False, True, True], [False, True, True, True]]},
        {"_name": "all padding", "valid": [[False, False, False]]},
        {"_name": "length 1", "valid": [[True], [False]]},
        {"_name": "zero-length sequences", "valid": [[], []]},
        {"_name": "empty batch", "valid": []},
        {"_name": "holes in the middle", "valid": [[True, False, True, False, True]]},
        {"_name": "random right-padded batch", "valid": right},
        {"_name": "random left-padded batch", "valid": left},
    ]


def build() -> Problem:
    return Problem(
        slug="causal_attention_mask",
        title="Build a causal attention mask",
        difficulty="easy",
        summary="Combine a causal mask with key and query padding into a (B, L, L) boolean attention mask.",
        params=["valid"],
        reference=reference,
        functional_cases=functional_cases,
        compare=compare,
        tags=["numpy", "attention", "masking", "transformer"],
    )
