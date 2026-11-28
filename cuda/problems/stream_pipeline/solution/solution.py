def solve(chunks, h2d, kernel, d2h):
    a, k, d = h2d / chunks, kernel / chunks, d2h / chunks
    free_in = free_sm = free_out = 0.0
    for _ in range(chunks):
        free_in = free_in + a
        free_sm = max(free_in, free_sm) + k
        free_out = max(free_sm, free_out) + d
    return round(free_out, 3)
