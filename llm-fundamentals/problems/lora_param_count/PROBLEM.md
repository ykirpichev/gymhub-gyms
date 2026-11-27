# Counting LoRA parameters

LoRA is applied to the query projection and the value projection in each of 24 layers. Each of those projections is a 4096×4096 matrix. The rank is r = 16 and alpha = 32. No biases are trained. The key projection, the output projection, and the feed-forward stay frozen.

A factor pair on a d×d matrix has `r×d + d×r` trainable parameters.

How many trainable LoRA parameters is that in total? Give the integer.
