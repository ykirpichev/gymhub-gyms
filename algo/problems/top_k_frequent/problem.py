import random
from collections import Counter

from gymkit.spec import Problem


def reference(words, k):
    counts = Counter(words)
    return sorted(counts, key=lambda word: (-counts[word], word))[:k]


_SYLLABLES = ["ka", "lo", "mi", "ne", "ru", "sa", "to", "vi", "ze", "po"]


def _random_case(seed, n, vocab, k, name):
    rng = random.Random(seed)
    pool = sorted({"".join(rng.choice(_SYLLABLES) for _ in range(rng.randint(1, 5))) for _ in range(vocab)})
    # Skewed weights so frequencies differ, with plenty of ties among rare words.
    weights = [1.0 / (rank + 1) for rank in range(len(pool))]
    rng.shuffle(weights)
    return {"_name": name, "words": rng.choices(pool, weights=weights, k=n), "k": min(k, len(pool))}


def functional_cases():
    return [
        {"_name": "example", "words": ["pear", "fig", "pear", "kiwi", "fig", "pear", "apple"], "k": 2},
        {"_name": "ties broken alphabetically", "words": ["b", "a", "c", "b", "a", "c"], "k": 2},
        {"_name": "k = number of distinct words", "words": ["x", "y", "y", "z", "z", "z"], "k": 3},
        {"_name": "single word repeated", "words": ["echo"] * 5, "k": 1},
        {"_name": "all distinct", "words": ["delta", "alpha", "charlie", "bravo"], "k": 3},
        {"_name": "prefixes sort first", "words": ["ab", "abc", "a", "abc", "ab", "a"], "k": 3},
        {"_name": "case matters", "words": ["Go", "go", "go", "Go", "GO"], "k": 3},
        _random_case(1, 300, 40, 10, "random n=300"),
        _random_case(2, 5000, 400, 50, "random n=5000"),
    ]


def performance_case(seed: int = 0):
    return _random_case(seed, 500_000, 20_000, 100, "perf")


def build() -> Problem:
    return Problem(
        slug="top_k_frequent",
        title="Top k frequent words",
        difficulty="easy",
        summary="The k most frequent words, ties broken alphabetically.",
        params=["words", "k"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        tags=["hash map", "sorting", "heap"],
    )
