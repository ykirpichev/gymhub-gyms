## Hint 1
The hinge term `max(0, 1 - m)` with margin `m = y · f(x)` has slope `-1` in `m` when `m < 1` and slope `0` when `m > 1`. At `m = 1` this problem fixes the slope at 0. A sample therefore pulls on `w` only while it is inside the margin or misclassified.

## Hint 2
The penalty `(λ/2)||w||²` adds `λw` to the gradient of `w` at every step, even for samples outside the margin. It adds nothing to `b`.

## Hint 3
Vectorize each step: `m = y * (X @ w + b)`, `active = m < 1`, `g_w = lam * w - (y[active] @ X[active]) / N`, `g_b = -y[active].sum() / N`. Update both parameters together, then compute the objective once at the end.
