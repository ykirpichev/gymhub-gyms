import random

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 1000, "cpp": 20}


def reference(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            if ca == cb:
                cur[j] = prev[j - 1]
            else:
                best = prev[j - 1]
                if prev[j] < best:
                    best = prev[j]
                if cur[j - 1] < best:
                    best = cur[j - 1]
                cur[j] = best + 1
        prev = cur
    return prev[-1]


def _mutate(rng, text, edits, alphabet):
    chars = list(text)
    for _ in range(edits):
        kind = rng.randrange(3)
        if kind == 0 or not chars:
            chars.insert(rng.randint(0, len(chars)), rng.choice(alphabet))
        elif kind == 1:
            chars.pop(rng.randrange(len(chars)))
        else:
            chars[rng.randrange(len(chars))] = rng.choice(alphabet)
    return "".join(chars)


def _related_case(seed, n, edits, alphabet, name):
    rng = random.Random(seed)
    a = "".join(rng.choice(alphabet) for _ in range(n))
    return {"_name": name, "a": a, "b": _mutate(rng, a, edits, alphabet)}


def _random_case(seed, n, m, alphabet, name):
    rng = random.Random(seed)
    return {
        "_name": name,
        "a": "".join(rng.choice(alphabet) for _ in range(n)),
        "b": "".join(rng.choice(alphabet) for _ in range(m)),
    }


def functional_cases():
    return [
        {"_name": "example", "a": "sunday", "b": "saturday"},
        {"_name": "both empty", "a": "", "b": ""},
        {"_name": "one empty", "a": "", "b": "abc"},
        {"_name": "other empty", "a": "hello", "b": ""},
        {"_name": "identical", "a": "gymnasium", "b": "gymnasium"},
        {"_name": "no common letters", "a": "abc", "b": "xyz"},
        {"_name": "repeated letters", "a": "aaaa", "b": "aa"},
        _random_case(1, 25, 18, "acgt", "random 25 x 18"),
        _related_case(2, 300, 40, "abcdefgh", "related 300 chars, 40 edits"),
        _random_case(3, 400, 250, "ab", "random 400 x 250, binary"),
    ]


def performance_case(seed: int = 0):
    return _related_case(seed, 2000, 400, "abcdefghijklmnopqrstuvwxyz", "perf")


def build() -> Problem:
    return Problem(
        slug="edit_distance",
        title="Edit distance",
        difficulty="medium",
        summary="Levenshtein distance between two strings with an O(n*m) DP and a rolling row.",
        params=["a", "b"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["dynamic programming", "strings"],
    )
