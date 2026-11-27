import torch

from gpu_gym.inputs import cuda_empty, cuda_rand
from gpu_gym.spec import Param, Problem


def reference(A, B, C, M, N, K):
    # FP16 times FP16 is exact in FP32, so the only error a kernel may add is from summing in FP32.
    C.copy_((A.double() @ B.double()).float())


def _case(m, n, k, seed, name):
    return {
        "_name": name,
        "A": cuda_rand(m, k, seed=seed, low=-1.0, high=1.0, dtype=torch.float16),
        "B": cuda_rand(k, n, seed=seed + 5, low=-1.0, high=1.0, dtype=torch.float16),
        "C": cuda_empty(m, n),
        "M": m,
        "N": n,
        "K": k,
    }


def functional_cases():
    # Every dimension is a multiple of 16, the size of one WMMA tile.
    shapes = [(16, 16, 16), (32, 48, 64), (64, 128, 32), (128, 64, 1024), (256, 256, 256), (1024, 1024, 4096)]
    return [_case(m, n, k, index + 1, f"{m}x{k} @ {k}x{n}") for index, (m, n, k) in enumerate(shapes)]


def performance_case(seed: int = 123):
    return _case(4096, 4096, 4096, seed, "perf")


def traffic(case):
    m, n, k = case["M"], case["N"], case["K"]
    return (m * k + k * n) * 2 + m * n * 4


def flops(case):
    return 2 * case["M"] * case["N"] * case["K"]


def build() -> Problem:
    return Problem(
        slug="gemm_fp16",
        title="FP16 matmul on Tensor Cores",
        difficulty="hard",
        summary="C = A @ B with FP16 inputs and an FP32 result: the shape Tensor Cores are built for.",
        # Allows a different FP32 summation order, and the reduced-precision accumulators some Tensor Cores use;
        # rejects accumulating in FP16, which loses about 3 digits at K = 4096.
        atol=2e-2,
        rtol=2e-3,
        params=[
            Param("A", "tensor", dtype="float16"),
            Param("B", "tensor", dtype="float16"),
            Param("C", "tensor", output=True),
            Param("M", "int"),
            Param("N", "int"),
            Param("K", "int"),
        ],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        traffic_bytes=traffic,
        flops=flops,
    )
