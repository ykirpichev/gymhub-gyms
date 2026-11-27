# A chain rule by hand

`x = torch.tensor(2.0, requires_grad=True)`, `y = x ** 3`, `z = torch.log(y) + y`, then `z.backward()`. What is `x.grad`? Give three decimal places.
