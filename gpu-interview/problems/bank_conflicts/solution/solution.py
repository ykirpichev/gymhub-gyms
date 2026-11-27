def solve(words):
    banks = {}
    for word in words:
        if word is None:
            continue
        banks.setdefault(word % 32, set()).add(word)
    return max((len(distinct) for distinct in banks.values()), default=0)
