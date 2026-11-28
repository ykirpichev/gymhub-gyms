def solve(lanes):
    v = list(lanes)
    for offset in (16, 8, 4, 2, 1):
        v = [v[i] + (v[i + offset] if i + offset < 32 else v[i]) for i in range(32)]
    return v
