from collections import Counter

END = "</w>"


def _apply(symbols, pair):
    merged, i = [], 0
    while i < len(symbols):
        if i + 1 < len(symbols) and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            merged.append(pair[0] + pair[1])
            i += 2
        else:
            merged.append(symbols[i])
            i += 1
    return merged


def solve(corpus, num_merges, words):
    if num_merges < 0:
        raise ValueError("num_merges must be >= 0")
    freq = Counter(corpus)
    split = {word: list(word) + [END] for word in freq}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word, symbols in split.items():
            for pair in zip(symbols, symbols[1:]):
                pairs[pair] += freq[word]
        if not pairs:
            break
        best = min(pairs, key=lambda p: (-pairs[p], p))  # most frequent, then smallest pair
        merges.append(best)
        split = {word: _apply(symbols, best) for word, symbols in split.items()}

    encodings = []
    for word in words:
        symbols = list(word) + [END]
        for pair in merges:
            symbols = _apply(symbols, pair)
        encodings.append(symbols)
    return merges, encodings
