# Active parameters in an MoE

An MoE decoder has 16 expert layers. Each of those layers has 8 experts, and each expert has 0.5 billion parameters. Every token is sent to its top 2 experts in each layer. Embeddings, attention, norms, and the routers add 1.5 billion parameters that every token uses.

How many billion parameters are active for one token? Give the number in billions (for example 12.5).
