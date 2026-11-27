# Optimizer memory for a 7B model

A 7-billion-parameter model trains with mixed-precision Adam: BF16 weights and gradients, an FP32 master copy of the weights and two FP32 Adam moments. How many **gigabytes** (10⁹ bytes) do weights, gradients and optimizer state take together, not counting activations?
