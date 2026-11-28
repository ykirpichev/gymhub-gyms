def solve(grid, block, n, mode):
    total = grid * block
    if mode == "stride":
        return [j % total for j in range(n)]
    return [j if j < total else -1 for j in range(n)]
