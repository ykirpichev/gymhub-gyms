def solve(shape, stride):
    def size(s):
        if isinstance(s, int):
            return s
        n = 1
        for m in s:
            n *= size(m)
        return n

    def offset(s, d, i):
        if isinstance(s, int):
            return i * d
        total = 0
        for sm, dm in zip(s, d):
            n = size(sm)
            total += offset(sm, dm, i % n)
            i //= n
        return total

    return [offset(shape, stride, i) for i in range(size(shape))]
