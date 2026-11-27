# GYM_STARTER
import torch


class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        pass

    @staticmethod
    def backward(ctx, grad_y):
        pass


def solve(x, grad_y):
    """Return (logsumexp over the last axis, gradient with respect to x)."""
    pass
