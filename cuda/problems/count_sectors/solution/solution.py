def solve(addresses, size):
    sectors, used = set(), set()
    for a in addresses:
        for s in range(a // 32, (a + size - 1) // 32 + 1):
            sectors.add(s)
        used.update(range(a, a + size))
    if not sectors:
        return [0, 0.0]
    return [len(sectors), round(len(used) / (32 * len(sectors)), 4)]
