# KV cache in GiB

Use this formula for the KV cache of one decoder:

`bytes = 2 × layers × kv_heads × head_dim × seq × batch × bytes_per_element`

The leading 2 counts keys and values. This deploy is 32 layers, 8 KV heads, head dimension 128, sequence length 2048, batch 4, and bf16 (2 bytes per element).

How many GiB is that cache? Use 1 GiB = 2^30 bytes. Give the number.
