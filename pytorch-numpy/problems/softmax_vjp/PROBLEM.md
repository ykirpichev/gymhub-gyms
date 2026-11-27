# Softmax backward by hand

For a row vector `x`, let `s = softmax(x)`. Given an upstream gradient `g = ∂L/∂s`, the gradient with respect to `x` is

$$\frac{\partial L}{\partial x} = s \odot \big(g - (g \cdot s)\big).$$

Write `solve(x, g)` for 2-D inputs (`N × C`, softmax over each row) that returns `(s, dx)` as NumPy arrays. Use a stable softmax. Don't build the `C × C` Jacobian.

```
solve([[0, 0]], [[1, 0]])  ->  (array([[0.5, 0.5]]), array([[0.25, -0.25]]))
```

#### What interviewers look for

- Deriving the formula from the Jacobian $J = \mathrm{diag}(s) - s s^\top$: $J^\top g = s \odot g - s (s \cdot g)$.
- The row-wise dot product with `keepdims=True` so it broadcasts.
- Noting that `dx` sums to zero in each row, because softmax is unchanged when you add a constant to `x`.

#### Follow-ups

- What is the backward of log-softmax? (`g - softmax(x) * g.sum()`.)
- Why do frameworks combine softmax and cross-entropy into one backward? (`dx = softmax(x) - onehot(y)`: simpler and stabler.)
- In a fused attention backward, which saved values let you skip storing `s`?
