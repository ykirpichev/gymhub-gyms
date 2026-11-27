import torch

from gymkit.spec import Problem


def _expert(v, wg, wu, wd):
    gate = v @ wg.T
    return (gate * torch.sigmoid(gate) * (v @ wu.T)) @ wd.T


def reference(x, router_w, w_gate, w_up, w_down, top_k):
    val, idx = torch.topk(x @ router_w.T, top_k, dim=-1)
    gates = torch.softmax(val, dim=-1)
    y = torch.zeros_like(x)
    for e in range(router_w.shape[0]):
        token, slot = (idx == e).nonzero(as_tuple=True)
        if token.numel():
            y.index_add_(0, token, _expert(x[token], w_gate[e], w_up[e], w_down[e]) * gates[token, slot, None])
    return y


def _case(t, d, f, e, k, seed, name, skew=False):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    router = randn(e, d)
    if skew:
        router[0] += 5.0 * randn(d).abs().sign()  # most tokens prefer expert 0; some experts get none
    return {
        "_name": name,
        "x": randn(t, d).abs() if skew else randn(t, d),
        "router_w": router,
        "w_gate": randn(e, f, d) / d ** 0.5,
        "w_up": randn(e, f, d) / d ** 0.5,
        "w_down": randn(e, d, f) / f ** 0.5,
        "top_k": k,
    }


def functional_cases():
    cuda = {"device": "cuda"}
    generator = torch.Generator(device="cuda").manual_seed(11)
    example = {
        "_name": "example",
        "x": torch.tensor([[1.0, 0.0]], **cuda),
        "router_w": torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]], **cuda),
        "w_gate": torch.randn(3, 4, 2, device="cuda", generator=generator),
        "w_up": torch.randn(3, 4, 2, device="cuda", generator=generator),
        "w_down": torch.randn(3, 2, 4, device="cuda", generator=generator),
        "top_k": 2,
    }
    return [
        example,
        _case(1, 8, 16, 4, 2, 1, "one token"),
        _case(16, 8, 16, 4, 1, 2, "top_k = 1"),
        _case(16, 8, 16, 4, 4, 3, "top_k = E: dense"),
        _case(64, 32, 64, 8, 2, 4, "E=8, k=2"),
        _case(64, 32, 64, 8, 2, 5, "skewed router: idle experts", skew=True),
        _case(256, 64, 96, 8, 3, 6, "T=256, k=3"),
    ]


def performance_case(seed: int = 0):
    return _case(4096, 1024, 2048, 8, 2, seed, "perf")


def build() -> Problem:
    return Problem(
        slug="moe_layer",
        title="Mixture-of-experts layer",
        difficulty="hard",
        summary="Top-k routing, per-expert SwiGLU MLPs on the tokens they receive, and a weighted combine.",
        params=["x", "router_w", "w_gate", "w_up", "w_down", "top_k"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-4,
        rtol=1e-3,
    )
