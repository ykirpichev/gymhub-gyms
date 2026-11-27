def solve(M, N, K, BM, BN, elem_bytes):
    if M % BM or N % BN:
        raise ValueError("tiles must divide the matrix")
    blocks = (M // BM) * (N // BN)
    bytes_read = blocks * (BM * K + K * BN) * elem_bytes
    flops = 2 * M * N * K
    return (bytes_read, flops / (bytes_read + M * N * elem_bytes))
