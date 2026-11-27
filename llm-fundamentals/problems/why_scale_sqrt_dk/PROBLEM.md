# Why scale the attention scores

A teammate deletes the `sqrt(d_k)` divisor in scaled dot-product attention "to save a broadcast." Query and key entries are roughly zero-mean with variance 1, and `d_k` is 128. A few hundred steps later the attention weights are almost one-hot and the gradient through the softmax has vanished.

Why does dividing by `sqrt(d_k)` prevent that?
