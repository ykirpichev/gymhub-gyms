import torch

from gymkit.spec import Problem


def _sample(r, u):
    cdf = torch.cumsum(r, dim=0)
    hits = (cdf > u * cdf[-1]).nonzero()
    return int(hits[0]) if hits.numel() else int(r.nonzero()[-1])


def reference(draft_tokens, draft_probs, target_probs, u_accept, u_sample):
    b_size, k = draft_tokens.shape
    accepted = torch.zeros(b_size, dtype=torch.int64, device=draft_tokens.device)
    nxt = torch.zeros(b_size, dtype=torch.int64, device=draft_tokens.device)
    for b in range(b_size):
        n = 0
        while n < k:
            x = int(draft_tokens[b, n])
            if float(u_accept[b, n] * draft_probs[b, n, x]) < float(target_probs[b, n, x]):
                n += 1
            else:
                break
        accepted[b] = n
        r = target_probs[b, k] if n == k else (target_probs[b, n] - draft_probs[b, n]).clamp(min=0)
        nxt[b] = _sample(r, u_sample[b])
    return accepted, nxt


def _case(b, k, v, seed, name, same=False, zero_target=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.softmax(2.0 * torch.randn(b, k, v, generator=generator), dim=-1)
    p = torch.softmax(2.0 * torch.randn(b, k + 1, v, generator=generator), dim=-1)
    if same:
        p[:, :k] = q
    tokens = torch.multinomial(q.reshape(b * k, v), 1, generator=generator).reshape(b, k)
    if zero_target:
        rows = torch.arange(b)
        p[rows, 0, tokens[:, 0]] = 0.0  # the target never says the first draft token
        p = p / p.sum(-1, keepdim=True)
    return {
        "_name": name,
        "draft_tokens": tokens.to("cuda"),
        "draft_probs": q.to("cuda"),
        "target_probs": p.to("cuda"),
        "u_accept": torch.rand(b, k, generator=generator).to("cuda"),
        "u_sample": torch.rand(b, generator=generator).to("cuda"),
    }


def functional_cases():
    cuda = {"device": "cuda"}
    example = {
        "_name": "example",
        "draft_tokens": torch.tensor([[0, 2]], dtype=torch.int64, **cuda),
        "draft_probs": torch.tensor([[[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]]], **cuda),
        "target_probs": torch.tensor([[[0.25, 0.5, 0.25], [0.5, 0.25, 0.25], [0.2, 0.3, 0.5]]], **cuda),
        "u_accept": torch.tensor([[0.4, 0.6]], **cuda),
        "u_sample": torch.tensor([0.7], **cuda),
    }
    return [
        example,
        _case(4, 3, 10, 1, "draft equals target: all accepted, bonus token", same=True),
        _case(6, 4, 10, 2, "target gives the first draft token probability 0", zero_target=True),
        _case(8, 1, 20, 3, "K = 1"),
        _case(16, 4, 50, 4, "random B=16, K=4, V=50"),
        _case(32, 5, 1000, 5, "random B=32, K=5, V=1000"),
    ]


def performance_case(seed: int = 0):
    return _case(256, 5, 32_000, seed, "perf")


def build() -> Problem:
    return Problem(
        slug="speculative_accept",
        title="Speculative decoding: accept and resample",
        difficulty="hard",
        summary="Batched verification of draft tokens with rejection sampling, so the output matches the target model exactly.",
        params=["draft_tokens", "draft_probs", "target_probs", "u_accept", "u_sample"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
    )
