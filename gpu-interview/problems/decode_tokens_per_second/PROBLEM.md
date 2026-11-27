# Tokens per second at batch size 1

A 7-billion-parameter model in BF16 generates text for one user. Each new token needs every weight read once from HBM (ignore the KV cache, activations and kernel launch overhead). On the interview GPU (3 TB/s), what is the most tokens per second it can generate?
