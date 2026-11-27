# KV cache per token with grouped-query attention

A model has 40 layers, 32 query heads and 8 key/value heads (grouped-query attention), a head dimension of 128, and keeps its KV cache in bf16 (2 bytes per value).

How many bytes of KV cache does **one token** take, across all layers? Give the integer.
