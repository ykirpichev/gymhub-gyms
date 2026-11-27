def solve(values):
    a = list(values)
    n = len(a)
    if n == 0 or n & (n - 1):
        raise ValueError("length must be a power of two")
    stride = 1
    while stride < n:
        for i in range(2 * stride - 1, n, 2 * stride):
            a[i] += a[i - stride]
        stride *= 2
    tree = list(a)
    a[n - 1] = 0
    stride = n // 2
    while stride >= 1:
        for i in range(2 * stride - 1, n, 2 * stride):
            t = a[i - stride]
            a[i - stride] = a[i]
            a[i] += t
        stride //= 2
    return (tree, a)
