# Size prefill with a prefix cache

A chat service at peak handles **200** messages a second on a **70B**-parameter model. Prefill costs about **2 × parameters** FLOPs per prompt token, and each GPU delivers **500 TFLOP/s** (5 × 10¹⁴ FLOP/s) of useful compute. With a prefix cache, only the **250** new tokens of each message need prefill. How many GPUs does prefill need? Give the exact quotient.
