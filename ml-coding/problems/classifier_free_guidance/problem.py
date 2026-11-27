"""Grader for classifier_free_guidance. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def _guide(cond, uncond, w):
    scale = float(w)
    if uncond is None:
        batch = np.asarray(cond, dtype=np.float64)
        if batch.ndim == 0 or batch.shape[0] % 2 != 0:
            raise ValueError("a stacked batch must have an even length on axis 0")
        half = batch.shape[0] // 2
        conditional = batch[:half]
        unconditional = batch[half:]
    else:
        conditional = np.asarray(cond, dtype=np.float64)
        unconditional = np.asarray(uncond, dtype=np.float64)
        if conditional.shape != unconditional.shape:
            raise ValueError("cond and uncond must have the same shape")
    return unconditional + scale * (conditional - unconditional)


def reference(cond, uncond, w):
    try:
        return _guide(cond, uncond, w)
    except ValueError:
        return "ValueError"


def call(module, args):
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    got_err = isinstance(got, str) and got == "ValueError"
    want_err = isinstance(want, str) and want == "ValueError"
    if got_err or want_err:
        if got_err != want_err:
            if want_err:
                raise AssertionError("expected ValueError (shapes differ, or the stacked batch has an odd axis-0 length)")
            raise AssertionError("raised ValueError on predictions that share a shape")
        return
    default_compare(got, want, 1e-9, 1e-7, "guided")


def _random(seed, shape, w, name, stacked=False):
    rng = np.random.default_rng(seed)
    if stacked:
        full = (shape[0] * 2,) + tuple(shape[1:])
        cond = rng.standard_normal(full).tolist()
        uncond = None
    else:
        cond = rng.standard_normal(shape).tolist()
        uncond = rng.standard_normal(shape).tolist()
    return {"_name": name, "cond": cond, "uncond": uncond, "w": float(w)}


def functional_cases():
    stacked = [[1.0, 2.0], [3.0, 4.0], [0.0, 1.0], [1.0, 0.0]]
    return [
        {"_name": "example", "cond": [1.0, 2.0, 3.0], "uncond": [0.0, 1.0, 1.0], "w": 2.0},
        {"_name": "w = 0 is unconditional", "cond": [1.0, 2.0, 3.0], "uncond": [0.0, 1.0, 1.0], "w": 0.0},
        {"_name": "w = 1 is conditional", "cond": [1.0, 2.0, 3.0], "uncond": [0.0, 1.0, 1.0], "w": 1.0},
        {"_name": "negative scale", "cond": [1.0, 2.0, 3.0], "uncond": [0.0, 1.0, 1.0], "w": -1.0},
        {"_name": "stacked batch", "cond": stacked, "uncond": None, "w": 2.0},
        {"_name": "stacked w = 0 returns the second half", "cond": stacked, "uncond": None, "w": 0.0},
        {"_name": "stacked w = 1 returns the first half", "cond": stacked, "uncond": None, "w": 1.0},
        {"_name": "rank 3",
         "cond": [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 2.0], [2.0, 2.0]]],
         "uncond": [[[0.0, 0.0], [0.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]],
         "w": 3.0},
        {"_name": "half interpolation", "cond": [[2.0, 4.0], [6.0, 8.0]], "uncond": [[0.0, 0.0], [2.0, 2.0]], "w": 0.5},
        _random(1, (4, 5), 2.5, "random (4, 5)"),
        _random(2, (3, 4, 4), 1.5, "random stacked (6, 4, 4)", stacked=True),
        _random(3, (8, 8), -0.3, "random (8, 8), negative w"),
        {"_name": "shapes differ", "cond": [[1.0], [2.0]], "uncond": [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], "w": 2.0},
        {"_name": "ranks differ", "cond": [1.0, 2.0, 3.0], "uncond": [[1.0, 2.0, 3.0]], "w": 1.0},
        {"_name": "stacked axis is odd", "cond": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], "uncond": None, "w": 1.0},
    ]


def build() -> Problem:
    return Problem(
        slug="classifier_free_guidance",
        title="Classifier-free guidance",
        difficulty="easy",
        summary="guided = uncond + w * (cond - uncond), including a batch stacked as conditional then unconditional.",
        params=["cond", "uncond", "w"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "diffusion", "guidance"],
    )
