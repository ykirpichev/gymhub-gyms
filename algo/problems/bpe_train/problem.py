import heapq
import random
from collections import Counter, defaultdict

from gymkit.spec import Problem


def reference(text, num_merges):
    word_freq = Counter(text.split())
    words = [list(word) for word in word_freq]
    freqs = list(word_freq.values())

    counts = defaultdict(int)  # pair -> weighted count
    where = defaultdict(set)  # pair -> ids of words that contain it
    for wid, symbols in enumerate(words):
        for pair in zip(symbols, symbols[1:]):
            counts[pair] += freqs[wid]
            where[pair].add(wid)

    heap = [(-count, pair) for pair, count in counts.items()]
    heapq.heapify(heap)
    merges = []
    while len(merges) < num_merges and heap:
        neg, pair = heapq.heappop(heap)
        if counts.get(pair, 0) != -neg or neg == 0:
            continue  # stale heap entry
        merges.append(pair)
        a, b = pair
        joined = a + b
        touched = {}
        for wid in list(where[pair]):
            symbols = words[wid]
            freq = freqs[wid]
            for old in zip(symbols, symbols[1:]):
                counts[old] -= freq
                touched[old] = counts[old]
            out = []
            i = 0
            while i < len(symbols):
                if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                    out.append(joined)
                    i += 2
                else:
                    out.append(symbols[i])
                    i += 1
            for old in set(zip(symbols, symbols[1:])):
                where[old].discard(wid)
            for new in zip(out, out[1:]):
                counts[new] += freq
                touched[new] = counts[new]
                where[new].add(wid)
            words[wid] = out
        for changed in touched:
            count = counts[changed]
            if count > 0:
                heapq.heappush(heap, (-count, changed))
            else:
                counts.pop(changed, None)
    return merges


def compare(got, want, case):
    if not isinstance(got, (list, tuple)):
        raise AssertionError(f"expected a list of merges, got {type(got).__name__}")
    normalized = []
    for index, item in enumerate(got):
        if not isinstance(item, (list, tuple)) or len(item) != 2 or not all(isinstance(s, str) for s in item):
            raise AssertionError(f"merge {index} should be a pair of strings, got {item!r}")
        normalized.append(tuple(item))
    want = [tuple(item) for item in want]
    for index, (g, w) in enumerate(zip(normalized, want)):
        if g != w:
            context = ", ".join(repr(m) for m in want[max(0, index - 3) : index])
            raise AssertionError(
                f"merge {index} is wrong: got {g!r}, expected {w!r}\n"
                f"  previous merges (all correct): [{context}]"
            )
    if len(normalized) != len(want):
        raise AssertionError(
            f"got {len(normalized)} merges, expected {len(want)}"
            + (f"; the next expected merge is {want[len(normalized)]!r}" if len(normalized) < len(want) else "")
        )


_CONSONANTS = "bcdfghklmnprstvwz"
_VOWELS = "aeiou"


def _corpus(seed, vocab_size, num_words):
    rng = random.Random(seed)
    syllables = [c + v + rng.choice(["", "", "n", "r", "s"]) for c in _CONSONANTS for v in _VOWELS]
    vocab = set()
    while len(vocab) < vocab_size:
        vocab.add("".join(rng.choice(syllables) for _ in range(rng.randint(1, 4))))
    vocab = sorted(vocab)
    rng.shuffle(vocab)
    weights = [1.0 / (rank + 1) for rank in range(len(vocab))]  # Zipf-like
    return " ".join(rng.choices(vocab, weights=weights, k=num_words))


def functional_cases():
    return [
        {"_name": "example", "text": "low low lower newest", "num_merges": 4},
        {"_name": "empty text", "text": "", "num_merges": 3},
        {"_name": "single-character words only", "text": "a b c a b", "num_merges": 5},
        {"_name": "runs out of pairs early", "text": "abc", "num_merges": 10},
        {"_name": "overlapping pair aaaa", "text": "aaaa aaa", "num_merges": 4},
        {"_name": "ties everywhere", "text": "ba ab cd dc", "num_merges": 4},
        {"_name": "whitespace and unicode", "text": "  héllo\thello\n\nhéllo  wörld  ", "num_merges": 6},
        {"_name": "small corpus", "text": _corpus(7, 300, 3000), "num_merges": 200},
    ]


def performance_case(seed: int = 0):
    return {"_name": "perf", "text": _corpus(seed, 20000, 300000), "num_merges": 2000}


def build() -> Problem:
    return Problem(
        slug="bpe_train",
        title="Train a BPE tokenizer",
        difficulty="medium",
        summary="Learn byte-pair-encoding merges from a corpus, updating pair counts incrementally.",
        params=["text", "num_merges"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        compare=compare,
        time_limit_ms=5000,
    )
