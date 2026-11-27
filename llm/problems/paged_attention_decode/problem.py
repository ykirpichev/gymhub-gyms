import math

import torch

from gymkit.spec import Problem


def reference(q, k_pool, v_pool, block_table, lengths):
    out = torch.empty_like(q)
    s = k_pool.shape[1]
    for b in range(q.shape[0]):
        n = int(lengths[b])
        t = torch.arange(n, device=q.device)
        blocks = block_table[b, t // s]
        keys = k_pool[blocks, t % s]  # [n, H, Dh]
        values = v_pool[blocks, t % s]
        scores = torch.einsum("hd,thd->ht", q[b], keys) / math.sqrt(q.shape[-1])
        out[b] = torch.einsum("ht,thd->hd", torch.softmax(scores, dim=-1), values)
    return out


def _case(b, h, dh, s, lengths, seed, name, share_prefix=False, junk=1e4):
    """Blocks are handed out in a scrambled order; slots past each sequence's length hold junk."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    blocks_needed = [-(-n // s) for n in lengths]
    m = max(blocks_needed)
    n_blocks = sum(blocks_needed) + 3
    order = torch.randperm(n_blocks, generator=generator).tolist()
    order.remove(0)  # block 0 stays free: padding entries point at it
    table = torch.zeros(b, m, dtype=torch.int64)
    for seq, count in enumerate(blocks_needed):
        for j in range(count):
            table[seq, j] = order.pop()
    if share_prefix:
        table[1, 0] = table[0, 0]  # sequence 1 reuses sequence 0's first block (a shared prompt prefix)
    k_pool = junk * torch.randn(n_blocks, s, h, dh, generator=generator)
    v_pool = junk * torch.randn(n_blocks, s, h, dh, generator=generator)
    for seq, n in enumerate(lengths):
        for t in range(n):
            blk, slot = int(table[seq, t // s]), t % s
            if share_prefix and seq == 1 and t < s:
                continue  # the shared block already holds sequence 0's prefix
            k_pool[blk, slot] = torch.randn(h, dh, generator=generator)
            v_pool[blk, slot] = torch.randn(h, dh, generator=generator)
    return {
        "_name": name,
        "q": torch.randn(b, h, dh, generator=generator).to("cuda"),
        "k_pool": k_pool.to("cuda"),
        "v_pool": v_pool.to("cuda"),
        "block_table": table.to("cuda"),
        "lengths": torch.tensor(lengths, dtype=torch.int64).to("cuda"),
    }


def functional_cases():
    cuda = {"device": "cuda"}
    example = {
        "_name": "example",
        "q": torch.tensor([[[1.0]], [[1.0]]], **cuda),
        "k_pool": torch.tensor([[[[0.0]], [[0.0]]], [[[1.0]], [[2.0]]], [[[3.0]], [[9.0]]]], **cuda),
        "v_pool": torch.tensor([[[[5.0]], [[5.0]]], [[[10.0]], [[20.0]]], [[[30.0]], [[99.0]]]], **cuda),
        "block_table": torch.tensor([[1, 2], [2, 0]], dtype=torch.int64, **cuda),
        "lengths": torch.tensor([3, 1], dtype=torch.int64, **cuda),
    }
    return [
        example,
        _case(1, 1, 4, 4, [3], 1, "one block"),
        _case(2, 2, 8, 4, [8, 4], 2, "lengths end on block boundaries"),
        _case(3, 2, 8, 4, [1, 1, 9], 3, "length 1"),
        _case(2, 4, 16, 8, [20, 13], 4, "shared prefix block", share_prefix=True),
        _case(4, 8, 64, 16, [100, 3, 64, 37], 5, "scrambled blocks, Dh=64"),
        _case(3, 4, 128, 16, [250, 17, 129], 6, "Dh=128"),
    ]


def performance_case(seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    lengths = torch.randint(256, 2048, (16,), generator=generator).tolist()
    return _case(16, 16, 128, 16, lengths, seed, "perf", junk=1.0)


def build() -> Problem:
    return Problem(
        slug="paged_attention_decode",
        title="Paged attention decode",
        difficulty="hard",
        summary="One decode step against a paged KV cache: block tables, a shared block pool, and per-sequence lengths.",
        params=["q", "k_pool", "v_pool", "block_table", "lengths"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-4,
        rtol=1e-4,
    )
