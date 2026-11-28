def solve(threads, regs, smem, max_warps, max_blocks, regs_per_sm, smem_per_sm):
    warps = (threads + 31) // 32
    regs_per_warp = (regs * 32 + 255) // 256 * 256
    limits = [max_warps // warps, max_blocks, (regs_per_sm // regs_per_warp) // warps]
    if smem:
        limits.append(smem_per_sm // smem)
    blocks = min(limits)
    return [blocks, round(blocks * warps / max_warps, 4)]
