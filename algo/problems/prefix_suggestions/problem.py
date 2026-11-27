import bisect
import random

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 400}


def reference(words, prefixes):
    vocab = sorted(set(words))
    out = []
    for prefix in prefixes:
        i = bisect.bisect_left(vocab, prefix)
        found = []
        while i < len(vocab) and len(found) < 3 and vocab[i].startswith(prefix):
            found.append(vocab[i])
            i += 1
        out.append(found)
    return out


SYLLABLES = ["ka", "ri", "mo", "ne", "to", "sa", "lu", "pe", "di", "o", "an", "ex"]


def _word(rng):
    return "".join(rng.choice(SYLLABLES) for _ in range(rng.randint(1, 5)))


def _random_case(seed, n_words, n_queries, name):
    rng = random.Random(seed)
    words = [_word(rng) for _ in range(n_words)]
    prefixes = []
    for _ in range(n_queries):
        if rng.random() < 0.7:
            w = rng.choice(words)
            prefixes.append(w[: rng.randint(0, len(w))])
        else:
            prefixes.append(_word(rng))
    return {"_name": name, "words": words, "prefixes": prefixes}


def functional_cases():
    return [
        {"_name": "example", "words": ["mouse", "mobile", "moneypot", "monitor", "mousepad", "money", "monitor"],
         "prefixes": ["mo", "mou", "mous", "x", ""]},
        {"_name": "empty dictionary", "words": [], "prefixes": ["a", ""]},
        {"_name": "no prefixes", "words": ["a", "b"], "prefixes": []},
        {"_name": "prefix equals a word", "words": ["app", "apple", "apply", "apt", "ape"], "prefixes": ["app", "ap"]},
        {"_name": "prefix longer than every word", "words": ["cat", "car"], "prefixes": ["cart", "catalog"]},
        {"_name": "duplicates count once", "words": ["bee", "bee", "bee", "been", "beer"], "prefixes": ["bee", "b"]},
        {"_name": "fewer than three", "words": ["zeta", "zebra"], "prefixes": ["z", "ze", "zeb"]},
        _random_case(1, 60, 30, "random 60 words"),
        _random_case(2, 2000, 500, "random 2000 words"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 100_000, 100_000, "perf")


def build() -> Problem:
    return Problem(
        slug="prefix_suggestions",
        title="Prefix suggestions",
        difficulty="medium",
        summary="Autocomplete: the three smallest words that start with each prefix, with a trie or a sorted list.",
        params=["words", "prefixes"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["trie", "binary search", "strings"],
    )
