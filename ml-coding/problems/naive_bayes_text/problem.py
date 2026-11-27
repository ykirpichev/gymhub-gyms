"""Grader for naive_bayes_text. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem

TIE_TOL = 1e-9


def reference(train_docs, train_labels, test_docs, alpha):
    if alpha <= 0:
        return "ValueError"
    classes = sorted(set(train_labels))
    vocab = sorted({t for doc in train_docs for t in doc})
    index = {t: j for j, t in enumerate(vocab)}
    counts = np.zeros((len(classes), len(vocab)))
    prior = np.zeros(len(classes))
    for doc, label in zip(train_docs, train_labels):
        k = classes.index(label)
        prior[k] += 1
        for t in doc:
            counts[k, index[t]] += 1
    log_prior = np.log(prior / prior.sum())
    log_lik = np.log((counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * len(vocab)))
    scores = np.zeros((len(test_docs), len(classes)))
    for i, doc in enumerate(test_docs):
        bag = np.zeros(len(vocab))
        for t in doc:
            if t in index:
                bag[index[t]] += 1
        scores[i] = log_prior + log_lik @ bag
    # Scores within TIE_TOL of the best are ties; the first class in sorted order wins.
    predictions = [classes[int(np.flatnonzero(row >= row.max() - TIE_TOL)[0])] for row in scores]
    return predictions, scores.reshape(len(test_docs), len(classes))


def call(module, args):
    """alpha <= 0 must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if got != want:
            raise AssertionError("expected ValueError for alpha <= 0" if want == "ValueError"
                                 else "raised ValueError, but alpha is positive")
        return
    if not isinstance(got, tuple) or len(got) != 2:
        raise AssertionError(f"return a tuple (predictions, scores), got {type(got).__name__}")
    scores = np.asarray(got[1], dtype=np.float64) if not isinstance(got[1], str) else None
    if scores is not None and scores.shape == want[1].shape and not np.all(np.isfinite(scores)):
        raise AssertionError("scores contain inf or NaN: multiplying many small probabilities underflows, "
                             "so add log-probabilities instead")
    default_compare(got[1], want[1], 1e-7, 1e-9, "scores")
    got_pred = list(got[0])
    if len(got_pred) != len(want[0]):
        raise AssertionError(f"predictions: expected {len(want[0])} labels, got {len(got_pred)}")
    for i, (g, w) in enumerate(zip(got_pred, want[0])):
        if g != w:
            raise AssertionError(f"predictions[{i}]: got {g!r}, expected {w!r} "
                                 "(highest score wins; on a tie, the class that sorts first)")


WORDS = ["ball", "goal", "team", "match", "vote", "law", "party", "court", "price", "market", "stock", "the", "a"]


def _random(seed, n_train, n_test, doc_len, name):
    rng = np.random.default_rng(seed)
    classes = ["business", "politics", "sport"]
    weights = rng.dirichlet(np.full(len(WORDS), 0.5), size=len(classes))
    labels = rng.choice(classes, size=n_train).tolist()
    docs = [rng.choice(WORDS, size=rng.integers(1, doc_len), p=weights[classes.index(c)]).tolist() for c in labels]
    test = [rng.choice(WORDS + ["unseen", "zzz"], size=rng.integers(0, doc_len)).tolist() for _ in range(n_test)]
    return {"_name": name, "train_docs": docs, "train_labels": labels, "test_docs": test, "alpha": 1.0}


def functional_cases():
    spam = [["win", "money", "now"], ["win", "prize"], ["meeting", "today"], ["lunch", "today", "now"]]
    spam_labels = ["spam", "spam", "ham", "ham"]
    return [
        {"_name": "example", "train_docs": spam, "train_labels": spam_labels,
         "test_docs": [["win", "now"], ["today"]], "alpha": 1.0},
        {"_name": "unseen tokens are ignored", "train_docs": spam, "train_labels": spam_labels,
         "test_docs": [["win", "bitcoin", "crypto"], ["unknown"], []], "alpha": 1.0},
        {"_name": "tie goes to the class that sorts first", "train_docs": [["a"], ["b"]],
         "train_labels": ["zeta", "alpha"], "test_docs": [[], ["c"], ["a", "b"], ["a"]], "alpha": 1.0},
        {"_name": "repeated tokens count", "train_docs": [["x", "x", "x", "y"], ["y", "y", "x"]],
         "train_labels": ["p", "q"], "test_docs": [["x", "x", "y"], ["x", "y", "y", "y"]], "alpha": 0.5},
        {"_name": "small alpha: smoothing still keeps unseen-in-class tokens finite",
         "train_docs": [["a", "b"], ["c"]], "train_labels": ["one", "two"],
         "test_docs": [["a", "c", "c"]], "alpha": 1e-3},
        {"_name": "large alpha flattens the likelihoods", "train_docs": spam, "train_labels": spam_labels,
         "test_docs": [["win", "now"], ["today"]], "alpha": 100.0},
        {"_name": "unequal priors", "train_docs": [["a"], ["a"], ["a"], ["b"]], "train_labels": ["x", "x", "x", "y"],
         "test_docs": [["b"], []], "alpha": 1.0},
        {"_name": "long document must not underflow", "train_docs": spam, "train_labels": spam_labels,
         "test_docs": [["win"] * 1500 + ["today"] * 1400, ["meeting", "now"] * 800], "alpha": 1.0},
        {"_name": "alpha 0 is invalid", "train_docs": spam, "train_labels": spam_labels,
         "test_docs": [["win"]], "alpha": 0.0},
        _random(1, 30, 10, 12, "random 30 docs"),
        _random(2, 300, 50, 40, "random 300 docs"),
    ]


def build() -> Problem:
    return Problem(
        slug="naive_bayes_text",
        title="Multinomial Naive Bayes for text",
        difficulty="medium",
        summary="Train a multinomial Naive Bayes text classifier with add-alpha smoothing and predict in log space.",
        params=["train_docs", "train_labels", "test_docs", "alpha"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["probability", "nlp", "smoothing"],
    )
