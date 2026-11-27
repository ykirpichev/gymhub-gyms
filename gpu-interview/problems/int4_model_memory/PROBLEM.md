# Memory for a 4-bit 70B model

A 70-billion-parameter model is quantized to 4 bits per weight, in groups of 128 weights, each group with one FP16 scale. How many **gigabytes** (10⁹ bytes) do the weights and scales take? Ignore the few layers left unquantized.
