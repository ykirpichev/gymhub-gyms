"""Grader for dpo_loss. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def _log_sigmoid(z):
    return np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))


def _loss(pc, pr, rc, rr, beta):
    return float(-np.mean(_log_sigmoid(beta * (np.asarray(pc, dtype=np.float64) - rc) - beta * (np.asarray(pr, dtype=np.float64) - rr))))


def reference(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta):
    arrays = [np.asarray(v, dtype=np.float64) for v in (policy_chosen, policy_rejected, ref_chosen, ref_rejected)]
    pc, pr, rc, rr = arrays
    if pc.ndim != 1 or pc.size == 0 or any(a.shape != pc.shape for a in arrays) or not beta > 0:
        return "ValueError"
    chosen = beta * (pc - rc)
    rejected = beta * (pr - rr)
    loss = float(-np.mean(_log_sigmoid(chosen - rejected)))
    return loss, chosen, rejected, float(np.mean(chosen > rejected))


def call(module, args):
    """Bad shapes or beta must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if got != want:
            raise AssertionError("expected ValueError: mismatched lengths, empty input or beta <= 0" if want == "ValueError"
                                 else "raised ValueError, but the inputs are valid")
        return
    if not isinstance(got, tuple) or len(got) != 4:
        raise AssertionError(f"return a tuple (loss, chosen_rewards, rejected_rewards, accuracy), got {type(got).__name__}")
    loss, chosen, rejected, accuracy = got
    default_compare(chosen, want[1], 1e-9, 1e-9, "chosen_rewards")
    default_compare(rejected, want[2], 1e-9, 1e-9, "rejected_rewards")
    if isinstance(loss, np.ndarray) and loss.size == 1:
        loss = loss.item()
    if isinstance(loss, (float, np.floating)) and not np.isfinite(loss):
        raise AssertionError(f"loss: got {loss}, expected {want[0]:.6g}. Large margins need a stable log-sigmoid.")
    try:
        default_compare(loss, want[0], 1e-9, 1e-9, "loss")
    except AssertionError as exc:
        args = [case[k] for k in ("policy_chosen", "policy_rejected", "ref_chosen", "ref_rejected")]
        swapped = _loss(args[1], args[0], args[3], args[2], case["beta"])
        if isinstance(loss, (float, int, np.floating)) and abs(loss - swapped) <= 1e-9 * max(1.0, abs(swapped)):
            raise AssertionError(f"{exc}\n  that is the loss with chosen and rejected swapped: z = chosen - rejected") from None
        raise
    default_compare(accuracy, want[3], 1e-12, 1e-12, "accuracy")


def _random(seed, n, beta, spread, name):
    rng = np.random.default_rng(seed)
    ref_c = -rng.uniform(5, 200, n)
    ref_r = -rng.uniform(5, 200, n)
    return {
        "_name": name,
        "policy_chosen": (ref_c + rng.normal(0.5, spread, n)).tolist(),
        "policy_rejected": (ref_r + rng.normal(-0.5, spread, n)).tolist(),
        "ref_chosen": ref_c.tolist(),
        "ref_rejected": ref_r.tolist(),
        "beta": beta,
    }


def functional_cases():
    return [
        {"_name": "example", "policy_chosen": [-12, -20, -8], "policy_rejected": [-15, -18, -9],
         "ref_chosen": [-13, -19, -8.5], "ref_rejected": [-14, -19, -8.5], "beta": 0.5},
        {"_name": "policy equals reference", "policy_chosen": [-5.0, -7.0], "policy_rejected": [-6.0, -4.0],
         "ref_chosen": [-5.0, -7.0], "ref_rejected": [-6.0, -4.0], "beta": 0.1},
        {"_name": "single pair, beta = 1", "policy_chosen": [-30.0], "policy_rejected": [-31.5],
         "ref_chosen": [-31.0], "ref_rejected": [-30.0], "beta": 1.0},
        {"_name": "huge positive margin", "policy_chosen": [-10.0], "policy_rejected": [-5000.0],
         "ref_chosen": [-3000.0], "ref_rejected": [-20.0], "beta": 1.0},
        {"_name": "huge negative margin", "policy_chosen": [-4000.0, -12.0], "policy_rejected": [-3.0, -11.0],
         "ref_chosen": [-10.0, -12.0], "ref_rejected": [-2000.0, -11.0], "beta": 0.5},
        {"_name": "large beta", "policy_chosen": [-100.0, -40.0, -55.0], "policy_rejected": [-90.0, -60.0, -50.0],
         "ref_chosen": [-101.0, -42.0, -50.0], "ref_rejected": [-88.0, -59.0, -52.0], "beta": 50.0},
        _random(1, 8, 0.1, 2.0, "random 8 pairs, beta 0.1"),
        _random(2, 2000, 0.25, 20.0, "random 2000 pairs, wide gaps"),
        {"_name": "length mismatch", "policy_chosen": [-1.0, -2.0], "policy_rejected": [-1.0],
         "ref_chosen": [-1.0, -2.0], "ref_rejected": [-1.0, -2.0], "beta": 0.1},
        {"_name": "beta = 0", "policy_chosen": [-1.0], "policy_rejected": [-2.0],
         "ref_chosen": [-1.0], "ref_rejected": [-2.0], "beta": 0.0},
    ]


def build() -> Problem:
    return Problem(
        slug="dpo_loss",
        title="Direct Preference Optimization loss",
        difficulty="hard",
        summary="DPO loss, implicit rewards and preference accuracy from policy and reference log-probabilities, stable for large margins.",
        params=["policy_chosen", "policy_rejected", "ref_chosen", "ref_rejected", "beta"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "alignment", "preference learning", "numerical stability"],
    )
