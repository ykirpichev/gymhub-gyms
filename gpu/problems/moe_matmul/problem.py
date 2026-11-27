import torch

from gpu_gym.inputs import cuda_empty, cuda_rand, cuda_tensor
from gpu_gym.spec import Param, Problem


def reference(X, expert, W, Y, T, E, K, N):
    ids = expert.long()
    x, w = X.double(), W.double()
    out = torch.zeros(T, N, dtype=torch.float64, device=X.device)
    for e in range(E):
        rows = (ids == e).nonzero(as_tuple=True)[0]
        if rows.numel():
            out[rows] = x[rows] @ w[e]
    Y.copy_(out.float())


def _experts(T, E, seed, skew=False):
    values = cuda_rand(T, seed=seed, low=0.0, high=1.0)
    if skew:
        # Most tokens go to expert 0: real routers are rarely balanced.
        values = values.pow_(4.0)
    values = (values * E).floor_()
    return values.clamp_(0.0, float(E - 1)).contiguous()


def _case(T, E, K, N, seed, name, skew=False):
    return {
        "_name": name,
        "X": cuda_rand(T, K, seed=seed, low=-1.0, high=1.0),
        "expert": _experts(T, E, seed + 3, skew),
        "W": cuda_rand(E, K, N, seed=seed + 7, low=-0.1, high=0.1),
        "Y": cuda_empty(T, N),
        "T": T,
        "E": E,
        "K": K,
        "N": N,
    }


def functional_cases():
    known = {
        "_name": "example",
        "X": cuda_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2)),
        "expert": cuda_tensor([1.0, 0.0, 1.0]),
        "W": cuda_tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0], (2, 2, 2)),
        "Y": cuda_empty(3, 2),
        "T": 3,
        "E": 2,
        "K": 2,
        "N": 2,
    }
    shapes = [(1, 1, 1, 1), (7, 3, 5, 4), (100, 4, 64, 32), (257, 8, 128, 96), (1000, 16, 33, 65)]
    random_cases = [
        _case(t, e, k, n, index + 1, f"T={t},E={e},K={k},N={n}")
        for index, (t, e, k, n) in enumerate(shapes)
    ]
    # Some experts get no tokens at all.
    empty = _case(5, 8, 16, 16, 31, "T=5,E=8: most experts empty")
    skewed = _case(2048, 8, 256, 256, 41, "T=2048,E=8, skewed routing", skew=True)
    large = _case(4096, 8, 512, 512, 99, "T=4096,E=8,K=512,N=512")
    return [known] + random_cases + [empty, skewed, large]


def performance_case(seed: int = 123):
    return _case(8192, 8, 1024, 1024, seed, "perf", skew=True)


def traffic(case):
    T, E, K, N = case["T"], case["E"], case["K"], case["N"]
    return (T * K + T + E * K * N + T * N) * 4


def flops(case):
    return 2 * case["T"] * case["K"] * case["N"]


def build() -> Problem:
    return Problem(
        slug="moe_matmul",
        title="MoE grouped matmul",
        difficulty="hard",
        summary="Multiply each token by the weight matrix of the expert it was routed to.",
        atol=1e-4,
        rtol=1e-4,
        params=[
            Param("X", "tensor"),
            Param("expert", "tensor"),
            Param("W", "tensor"),
            Param("Y", "tensor", output=True),
            Param("T", "int"),
            Param("E", "int"),
            Param("K", "int"),
            Param("N", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
        flops=flops,
    )
