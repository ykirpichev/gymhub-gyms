# Plan a fused kernel

An LLM layer runs RMSNorm on BF16 activations of shape `[tokens, 8192]`, then quantizes the result to FP8 E4M3 with one
scale per row for the next GEMM. Today these are two PyTorch ops. Describe the single CUDA (or Triton) kernel you would
write: how work maps to blocks, warps and threads, how the row's statistics are computed, how the FP8 scale is chosen,
what memory traffic the fused kernel has compared with the two separate ops, and how you would test and benchmark it.
