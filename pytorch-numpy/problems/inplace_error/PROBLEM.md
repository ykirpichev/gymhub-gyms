# An in-place error

```python
x = torch.randn(4, requires_grad=True)
y = torch.sigmoid(x)      # line A
y.mul_(2)                 # line B
loss = y.sum()            # line C
loss.backward()           # line D
```

Line D raises *"one of the variables needed for gradient computation has been modified by an inplace operation"*. Why?
