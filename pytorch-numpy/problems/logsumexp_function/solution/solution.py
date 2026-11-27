import torch


class LogSumExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        m = x.max(dim=-1, keepdim=True).values
        y = (m + torch.log(torch.exp(x - m).sum(dim=-1, keepdim=True))).squeeze(-1)
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, y = ctx.saved_tensors
        return grad_y.unsqueeze(-1) * torch.exp(x - y.unsqueeze(-1))


def solve(x, grad_y):
    t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y = LogSumExp.apply(t)
    y.backward(torch.tensor(grad_y, dtype=torch.float64))
    return (y.detach(), t.grad.detach())
