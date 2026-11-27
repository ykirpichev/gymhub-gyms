"""Grader for byte_pair_encoding. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import random
from collections import Counter

from gymkit.spec import Problem

END = "</w>"


def _merge(symbols, pair):
    out, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return out


def reference(corpus, num_merges, words):
    if num_merges < 0:
        return "ValueError"
    vocab = {word: list(word) + [END] for word in set(corpus)}
    freq = Counter(corpus)
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, symbols in vocab.items():
            for pair in zip(symbols, symbols[1:]):
                pairs[pair] += freq[word]
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))
        merges.append(best)
        vocab = {word: _merge(symbols, best) for word, symbols in vocab.items()}
    encodings = []
    for word in words:
        symbols = list(word) + [END]
        for pair in merges:
            symbols = _merge(symbols, pair)
        encodings.append(symbols)
    return [list(p) for p in merges], encodings


def call(module, args):
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError("expected ValueError for num_merges < 0" if want == "ValueError"
                                 else "raised ValueError, but num_merges is valid")
        return
    if not isinstance(got, tuple) or len(got) != 2:
        raise AssertionError(f"return a tuple (merges, encodings), got {type(got).__name__}")
    merges = [list(p) if isinstance(p, (tuple, list)) else p for p in got[0]]
    for i, (g, w) in enumerate(zip(merges, want[0])):
        if g != w:
            raise AssertionError(f"merge #{i}: got {g!r}, expected {w!r} (merges so far: {want[0][:i]!r:.200})")
    if len(merges) != len(want[0]):
        raise AssertionError(f"learned {len(merges)} merges, expected {len(want[0])}")
    encodings = [list(e) for e in got[1]]
    if len(encodings) != len(want[1]):
        raise AssertionError(f"got {len(encodings)} encodings, expected one per word ({len(want[1])})")
    for word, g, w in zip(case["words"], encodings, want[1]):
        if g != w:
            raise AssertionError(f"encoding of {word!r}: got {g!r}, expected {w!r}")


def _random(seed, n_words, alphabet, max_len, num_merges, n_new, name):
    rng = random.Random(seed)
    stems = ["".join(rng.choice(alphabet) for _ in range(rng.randint(1, max_len))) for _ in range(n_words // 4 + 1)]
    corpus = [rng.choice(stems) + rng.choice(["", "s", "ed", "ing"]) for _ in range(n_words)]
    words = ["".join(rng.choice(alphabet + "xyz") for _ in range(rng.randint(0, max_len + 2))) for _ in range(n_new)]
    return {"_name": name, "corpus": corpus, "num_merges": num_merges, "words": words + corpus[:3]}


def functional_cases():
    return [
        {"_name": "example", "corpus": ["low", "low", "lower", "newest", "newest", "widest"], "num_merges": 4,
         "words": ["lowest", "new", "slow"]},
        {"_name": "zero merges", "corpus": ["ab", "ab"], "num_merges": 0, "words": ["ab", ""]},
        {"_name": "ties break by the smaller pair", "corpus": ["ba", "dc"], "num_merges": 3, "words": ["bad", "dcba"]},
        {"_name": "frequency weighting", "corpus": ["xy", "ab", "xy"], "num_merges": 2, "words": ["abxy", "xyz"]},
        {"_name": "overlapping pairs merge left to right", "corpus": ["aaaa", "aaa"], "num_merges": 3,
         "words": ["aaaaa", "aa", "a"]},
        {"_name": "runs out of pairs", "corpus": ["ab"], "num_merges": 10, "words": ["ab", "abab", "ba"]},
        {"_name": "empty corpus", "corpus": [], "num_merges": 5, "words": ["hi", ""]},
        {"_name": "unseen characters stay single", "corpus": ["hug", "hug", "pug", "hugs"], "num_merges": 5,
         "words": ["bug", "hugz", "ü"]},
        {"_name": "negative num_merges", "corpus": ["ab"], "num_merges": -1, "words": []},
        _random(1, 40, "abcde", 6, 20, 10, "random small"),
        _random(2, 2000, "abcdefghij", 8, 150, 200, "random larger"),
    ]


def build() -> Problem:
    return Problem(
        slug="byte_pair_encoding",
        title="Learn and apply byte-pair encoding",
        difficulty="easy",
        summary="Learn BPE merges from a word corpus with deterministic tie-breaking, then encode new words by replaying them.",
        params=["corpus", "num_merges", "words"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["tokenization", "nlp", "strings"],
    )
