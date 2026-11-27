# Chinchilla compute

Use two rules of thumb for a dense transformer. Compute-optimal training uses about 20 tokens per parameter. Training compute is about `C = 6ND` FLOPs, where `N` is the parameter count and `D` is the number of training tokens.

For `N = 5 × 10^9`, how many FLOPs is that compute-optimal run? Scientific notation such as `1e21` is fine.
