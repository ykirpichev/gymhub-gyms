"""Grader for sinusoidal_position_encoding. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def reference(seq_len, d_model):
    if seq_len < 0 or d_model < 1:
        return "ValueError"
    c = np.arange(d_model)
    angle = np.arange(seq_len)[:, None] / np.power(10000.0, (c - c % 2) / d_model)
    return np.where(c % 2 == 0, np.sin(angle), np.cos(angle))


def call(module, args):
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if got != want:
            raise AssertionError("expected ValueError (seq_len < 0 or d_model < 1)" if isinstance(want, str)
                                 else "raised ValueError, but the arguments are valid")
        return
    g = np.asarray(got)
    if g.shape == want.shape and g.size:
        bad = ~np.isclose(g, want, atol=1e-9, rtol=1e-7)
        if bad.any():
            pos, c = (int(x) for x in np.argwhere(bad)[0])
            kind = "sin" if c % 2 == 0 else "cos"
            raise AssertionError(
                f"{int(bad.sum())} of {want.size} values wrong; first at [pos={pos}, channel={c}]: got {g[pos, c]:.9g}, "
                f"expected {want[pos, c]:.9g} = {kind}({pos} / 10000^({c - c % 2}/{case['d_model']}))")
    default_compare(got, want, 1e-9, 1e-7, "pe")


def functional_cases():
    return [
        {"_name": "example", "seq_len": 3, "d_model": 4},
        {"_name": "odd width ends with sin", "seq_len": 4, "d_model": 5},
        {"_name": "d_model 1", "seq_len": 3, "d_model": 1},
        {"_name": "d_model 2", "seq_len": 5, "d_model": 2},
        {"_name": "seq_len 0", "seq_len": 0, "d_model": 6},
        {"_name": "seq_len 1 (position 0 only)", "seq_len": 1, "d_model": 7},
        {"_name": "typical size", "seq_len": 128, "d_model": 64},
        {"_name": "long and wide, odd", "seq_len": 2048, "d_model": 513},
        {"_name": "negative seq_len", "seq_len": -1, "d_model": 4},
        {"_name": "d_model 0", "seq_len": 3, "d_model": 0},
    ]


def build() -> Problem:
    return Problem(
        slug="sinusoidal_position_encoding",
        title="Sinusoidal positional encoding",
        difficulty="easy",
        summary="The (L, d) Transformer position matrix with sine on even channels and cosine on odd ones, for any width.",
        params=["seq_len", "d_model"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "transformer", "vectorization"],
    )
