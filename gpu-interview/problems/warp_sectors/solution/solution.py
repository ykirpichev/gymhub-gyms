def solve(addresses, size):
    sectors = set()
    wanted = set()
    for address in addresses:
        if address is None:
            continue
        first, last = address // 32, (address + size - 1) // 32
        sectors.update(range(first, last + 1))
        wanted.update(range(address, address + size))
    if not sectors:
        return (0, 0.0)
    return (len(sectors), len(wanted) / (32 * len(sectors)))
