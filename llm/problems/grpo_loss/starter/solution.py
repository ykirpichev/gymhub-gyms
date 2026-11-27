# GYM_STARTER
import torch


def solve(logp_new: torch.Tensor, logp_old: torch.Tensor, logp_ref: torch.Tensor, rewards: torch.Tensor,
          mask: torch.Tensor, group_size: int, clip_eps: float, beta: float) -> tuple[torch.Tensor, torch.Tensor]:
    pass
