def solve(words):
    banks = {}
    for w in words:
        banks.setdefault(w % 32, set()).add(w)
    return max((len(s) for s in banks.values()), default=0)
