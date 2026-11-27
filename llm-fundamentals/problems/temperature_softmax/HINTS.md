## Hint 1
Divide the logits by 2 first. Then softmax is exp(z_i) over the sum of exp(z) across all three tokens.
