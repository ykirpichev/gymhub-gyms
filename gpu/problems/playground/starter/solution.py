import torch


def solve(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, N: int) -> None:
    torch.add(A, B, out=C)
