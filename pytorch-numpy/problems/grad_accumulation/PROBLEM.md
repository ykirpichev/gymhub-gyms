# Gradients accumulate

```python
x = torch.tensor(3.0, requires_grad=True)
(x ** 2).backward()
(x * 5).backward()
```

What is `x.grad` now?
