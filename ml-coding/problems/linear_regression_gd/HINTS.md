## Hint 1
Write the residual `r = X·w + b − y` (shape `(N,)`). The loss is `mean(r²)`, and every gradient is a function of `r`.

## Hint 2
Differentiate `(1/N) Σ rᵢ²`: `∂L/∂w = (2/N) Xᵀ r` and `∂L/∂b = (2/N) Σ rᵢ`. Compute both from the same `r` before changing either parameter.

## Hint 3
Loop `steps` times: compute `r`, append `mean(r²)`, update `w` and `b`. After the loop, append the loss once more for the final parameters, which gives `steps + 1` values.
