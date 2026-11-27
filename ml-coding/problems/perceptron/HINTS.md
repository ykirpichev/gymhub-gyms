## Hint 1
At the start `w = 0` and `b = 0`, so every score is 0. If your condition is `< 0`, the perceptron never makes its first update.

## Hint 2
The update is online: each mistake changes `w` and `b` immediately, and the next sample is scored with the new values. Do not compute all scores up front for the epoch.

## Hint 3
Two nested loops: over epochs, then over samples. Inside, compute `y[i] * (X[i] @ w + b)`; if it is `<= 0`, add `y[i] * X[i]` to `w`, add `y[i]` to `b` and count the update.
