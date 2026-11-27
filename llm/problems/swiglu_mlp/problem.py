import torch

from gymkit.spec import Problem


def reference(x, w_gate, w_up, w_down):
    gate = x @ w_gate.T
    return (gate * torch.sigmoid(gate) * (x @ w_up.T)) @ w_down.T


def _case(lead, d, f, seed, name, scale=1.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, device="cuda", generator=generator)

    return {
        "_name": name,
        "x": scale * randn(*lead, d),
        "w_gate": randn(f, d) / d ** 0.5,
        "w_up": randn(f, d) / d ** 0.5,
        "w_down": randn(d, f) / f ** 0.5,
    }


def functional_cases():
    cuda = {"device": "cuda"}
    example = {
        "_name": "example",
        "x": torch.tensor([[1.0, -1.0]], **cuda),
        "w_gate": torch.tensor([[1.0, 0.0], [0.0, 1.0]], **cuda),
        "w_up": torch.tensor([[2.0, 0.0], [0.0, 2.0]], **cuda),
        "w_down": torch.tensor([[1.0, 1.0], [0.0, 1.0]], **cuda),
    }
    return [
        example,
        _case((1,), 1, 1, 1, "D=1, F=1"),
        _case((3,), 8, 20, 2, "one leading dim"),
        _case((2, 3, 5), 16, 40, 3, "three leading dims"),
        _case((4, 33), 128, 352, 4, "odd sizes"),
        _case((2, 64), 512, 512, 5, "D=F=512"),
        _case((2, 7), 32, 64, 6, "large activations", scale=200.0),
    ]


def performance_case(seed: int = 0):
    return _case((4, 512), 2048, 5632, seed, "perf")


def build() -> Problem:
    return Problem(
        slug="swiglu_mlp",
        title="SwiGLU feed-forward block",
        difficulty="easy",
        summary="The gated MLP of Llama-style transformers: two projections, SiLU gate, and a down projection.",
        params=["x", "w_gate", "w_up", "w_down"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-3,
        rtol=1e-3,
    )
