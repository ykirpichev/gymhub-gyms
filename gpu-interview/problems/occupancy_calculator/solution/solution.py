def solve(threads_per_block, regs_per_thread, smem_per_block, sm):
    if threads_per_block < 1 or threads_per_block > sm["max_threads_per_block"]:
        raise ValueError("bad block size")
    warps = (threads_per_block + 31) // 32
    threads_alloc = warps * 32
    regs_per_block = regs_per_thread * threads_alloc
    if regs_per_block > sm["registers"] or smem_per_block > sm["shared"]:
        raise ValueError("one block does not fit on an SM")
    limits = [
        (sm["max_threads"] // threads_alloc, "threads"),
        (sm["max_blocks"], "blocks"),
        (sm["registers"] // regs_per_block if regs_per_block else float("inf"), "registers"),
        (sm["shared"] // smem_per_block if smem_per_block else float("inf"), "shared"),
    ]
    blocks = min(count for count, _ in limits)
    limiter = next(name for count, name in limits if count == blocks)
    occupancy = blocks * warps / (sm["max_threads"] / 32)
    return (int(blocks), occupancy, limiter)
