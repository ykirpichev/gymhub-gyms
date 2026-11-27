import torch


def solve(X, y, w0, b0, lr, steps):
    X = torch.tensor(X, dtype=torch.float64)
    y = torch.tensor(y, dtype=torch.float64)
    w = torch.tensor(w0, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(float(b0), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=lr)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = ((X @ w + b - y) ** 2).mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return (w.detach(), b.item(), losses)
