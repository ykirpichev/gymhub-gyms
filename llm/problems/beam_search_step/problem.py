import torch

from gymkit.spec import Problem


def reference(beam_scores, logprobs):
    b, w, v = logprobs.shape
    out_scores = torch.empty(b, w, device=logprobs.device)
    parents = torch.empty(b, w, dtype=torch.int64, device=logprobs.device)
    tokens = torch.empty(b, w, dtype=torch.int64, device=logprobs.device)
    for i in range(b):
        cand = (beam_scores[i][:, None] + logprobs[i]).flatten()
        order = torch.argsort(cand, descending=True, stable=True)[:w]
        out_scores[i] = cand[order]
        parents[i] = order // v
        tokens[i] = order % v
    return out_scores, parents, tokens


def _case(b, w, v, seed, name, spread=1.0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logprobs = torch.log_softmax(2.0 * torch.randn(b, w, v, generator=generator), dim=-1)
    beams = -spread * torch.rand(b, w, generator=generator) * 5.0
    return {"_name": name, "beam_scores": beams.to("cuda"), "logprobs": logprobs.to("cuda")}


def functional_cases():
    cuda = {"device": "cuda"}
    example = {
        "_name": "example",
        "beam_scores": torch.tensor([[-1.0, -2.0]], **cuda),
        "logprobs": torch.tensor([[[-0.1, -2.0, -3.0], [-0.2, -0.3, -4.0]]], **cuda),
    }
    return [
        example,
        _case(3, 1, 10, 1, "W = 1: greedy"),
        _case(2, 4, 1, 2, "V = 1: every beam survives"),
        _case(1, 4, 50, 3, "one input"),
        _case(4, 3, 20, 4, "one dominant beam", spread=40.0),
        _case(8, 5, 300, 5, "random B=8, W=5, V=300"),
    ]


def performance_case(seed: int = 0):
    return _case(64, 8, 32_000, seed, "perf")


def build() -> Problem:
    return Problem(
        slug="beam_search_step",
        title="Beam search step",
        difficulty="medium",
        summary="Extend W beams by one token for a whole batch: top-W over W·V candidates, with parents and tokens.",
        params=["beam_scores", "logprobs"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        atol=1e-5,
        rtol=1e-5,
    )
