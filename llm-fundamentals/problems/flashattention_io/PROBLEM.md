# What FlashAttention changes

Long-context training dies because each layer writes an N×N attention matrix into GPU memory. A colleague describes FlashAttention as "a faster approximate attention, like a local window."

Which description is right?
