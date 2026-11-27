# One pre-norm block, in order

A pre-norm decoder block, as in Llama-style models, computes

$$h = x + \mathrm{Attn}(\mathrm{Norm}(x)), \qquad y = h + \mathrm{MLP}(\mathrm{Norm}(h))$$

Put the six operations in the order a token's vector meets them, first at the top.
