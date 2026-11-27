import torch

from gymkit.spec import Problem

# A token whose cumulative probability (before it) lies this close to top_p may go either way
# depending on summation order, so the checker accepts both answers for it.
BORDER = 1e-5


def _top_k(logits, top_k):
    if 0 < top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return logits


def _cum_before(logits):
    """Probability mass of all strictly more likely tokens, per token, in float64."""
    probs = torch.softmax(logits.double(), dim=-1)
    sorted_probs, order = torch.sort(probs, dim=-1, descending=True)
    before_sorted = torch.cumsum(sorted_probs, dim=-1) - sorted_probs
    before_sorted[..., 0] = 0.0
    return torch.empty_like(before_sorted).scatter_(-1, order, before_sorted)


def reference(logits, top_k, top_p):
    out = _top_k(logits, top_k)
    if top_p < 1.0:
        before = _cum_before(out)
        drop = before >= top_p
        top1 = out.argmax(dim=-1, keepdim=True)
        drop.scatter_(-1, top1, False)
        out = out.masked_fill(drop, float("-inf"))
    return out


def compare(got, want, case):
    if not isinstance(got, torch.Tensor):
        raise AssertionError(f"expected a tensor, got {type(got).__name__}")
    if tuple(got.shape) != tuple(want.shape):
        raise AssertionError(f"shape {list(got.shape)} != expected {list(want.shape)}")
    if got.dtype != want.dtype:
        raise AssertionError(f"dtype {got.dtype} != expected {want.dtype}")
    g = got.detach().cpu()
    w = want.detach().cpu()
    if bool(torch.isnan(g).any()):
        raise AssertionError("the output contains NaN")
    if bool(torch.isposinf(g).any()):
        raise AssertionError("the output contains +inf; filtered entries must be -inf")
    g_drop, w_drop = torch.isneginf(g), torch.isneginf(w)

    both_kept = ~g_drop & ~w_drop
    changed = both_kept & ~torch.isclose(g, w, atol=1e-6, rtol=1e-6)
    if bool(changed.any()):
        lines = [f"{int(changed.sum())} kept logits were modified; kept entries must be returned unchanged"]
        for b, v in changed.nonzero()[:5].tolist():
            lines.append(f"  [{b}, {v}]  got {g[b, v].item():.6g}  expected {w[b, v].item():.6g}")
        raise AssertionError("\n".join(lines))

    differ = g_drop != w_drop
    if bool(differ.any()) and case["top_p"] < 1.0:
        before = _cum_before(_top_k(case["logits"], case["top_k"])).cpu()
        borderline = (before - case["top_p"]).abs() < BORDER
        differ &= ~borderline
    if bool(differ.any()):
        rows = differ.any(dim=-1).nonzero().flatten().tolist()
        lines = [f"{int(differ.sum())} entries are filtered wrongly, in {len(rows)} of {w.shape[0]} rows"]
        for b in rows[:3]:
            kept_g = int((~g_drop[b]).sum())
            kept_w = int((~w_drop[b]).sum())
            lines.append(f"  row {b}: you keep {kept_g} tokens, expected {kept_w}")
            for v in differ[b].nonzero().flatten()[:3].tolist():
                state = "removed" if g_drop[b, v] else "kept"
                lines.append(f"    token {v} (logit {w[b, v].item() if not w_drop[b, v] else g[b, v].item():.4f}): you {state} it")
        raise AssertionError("\n".join(lines))


def _case(b, v, top_k, top_p, seed, name, scale=3.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = scale * torch.randn(b, v, device="cuda", generator=generator)
    return {"_name": name, "logits": logits, "top_k": top_k, "top_p": top_p}


def functional_cases():
    example = {
        "_name": "example",
        "logits": torch.tensor([[2.0, 1.0, 0.0, -1.0, 3.0]], device="cuda"),
        "top_k": 4,
        "top_p": 0.8,
    }
    return [
        example,
        _case(4, 100, 0, 1.0, 1, "both disabled (identity)"),
        _case(8, 1000, 10, 1.0, 2, "top_k only"),
        _case(8, 1000, 0, 0.9, 3, "top_p only"),
        _case(16, 5000, 50, 0.7, 4, "top_k then top_p"),
        _case(4, 300, 1, 0.9, 5, "top_k=1 (greedy)"),
        _case(4, 64, 100, 0.95, 6, "top_k larger than V"),
        _case(8, 2000, 0, 1e-6, 7, "tiny top_p keeps only the top token"),
    ]


def performance_case(seed: int = 0):
    return _case(64, 128256, 1000, 0.95, seed, "perf")


def build() -> Problem:
    return Problem(
        slug="sampling_filter",
        title="Top-k / top-p filtering",
        difficulty="medium",
        summary="Mask logits outside the top-k set and the nucleus (top-p) set before sampling.",
        params=["logits", "top_k", "top_p"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        compare=compare,
    )
