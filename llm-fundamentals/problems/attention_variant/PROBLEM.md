# Picking an attention variant

A serving engineer is choosing an attention variant for a long-context decoder. A pilot of multi-query attention, one key/value head shared by every query head, lost too much quality. They want fewer key/value heads than query heads, but more than one, and they do not want a low-rank latent in place of the keys and values.

Which variant is that?
