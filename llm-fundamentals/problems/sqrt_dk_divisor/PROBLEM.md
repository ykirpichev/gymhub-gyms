# The attention divisor

A decoder has model width `d_model = 4096`, split evenly across 32 attention heads. Scaled dot-product attention divides each head's query-key scores by `sqrt(d_k)` before the softmax.

What is that divisor? Give the number to two decimal places.
