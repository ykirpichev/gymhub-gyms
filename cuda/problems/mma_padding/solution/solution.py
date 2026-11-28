def solve(M, N, K, m, n, k):
    ceil = lambda a, b: (a + b - 1) // b
    count = ceil(M, m) * ceil(N, n) * ceil(K, k)
    return [count, round(M * N * K / (count * m * n * k), 4)]
