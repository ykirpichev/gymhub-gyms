def solve(flops, bytes, peak_tflops, bw_tbs):
    compute = flops / (peak_tflops * 1e12) * 1e6
    memory = bytes / (bw_tbs * 1e12) * 1e6
    if compute > memory:
        return [round(compute, 2), "compute"]
    return [round(memory, 2), "memory"]
