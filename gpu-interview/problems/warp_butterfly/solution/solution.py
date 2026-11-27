def solve(values, steps):
    lanes = list(values)
    offset = 16
    for _ in range(steps):
        lanes = [lanes[i] + lanes[i ^ offset] for i in range(32)]
        offset //= 2
    return lanes
