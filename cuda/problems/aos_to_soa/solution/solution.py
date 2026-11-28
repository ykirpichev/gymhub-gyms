def solve(aos, fields):
    soa = [aos[f::fields] for f in range(fields)]
    return [soa, fields, 1]
