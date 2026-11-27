# Numerics and speed in one page

**Defaults differ.** NumPy creates `float64` arrays from Python floats; PyTorch creates `float32` tensors. Moving code between them changes results in the 7th digit, and comparing them with `==` fails. Compare with a tolerance (`np.allclose`, `torch.testing.assert_close`).

**Stable formulas** avoid overflow and cancellation: subtract the max before `exp` (softmax, log-sum-exp); use `log1p(x)` for `log(1 + x)` and `expm1` near zero; compute the binary cross-entropy from logits (`BCEWithLogitsLoss`), not from probabilities; use Welford's update or a two-pass formula for variance, not `E[x²] − E[x]²`.

**Summation order matters** in floating point: `(a + b) + c` can differ from `a + (b + c)`. A long sum in `float32` drifts; summing in pairs (as NumPy does) or with Kahan compensation keeps the error small.

**Mixed precision.** `torch.autocast` runs matmuls and convolutions in FP16 or BF16 and keeps reductions and losses in FP32. With FP16, a `GradScaler` multiplies the loss so small gradients don't underflow, then unscales before the optimizer step; BF16 has FP32's range and needs no scaler.

**Randomness.** `torch.manual_seed` seeds PyTorch's generators; data loader workers, NumPy and Python's `random` need their own seeds. Even with seeds, some GPU kernels (atomics in scatter-add, some backward passes) are non-deterministic unless you ask for `torch.use_deterministic_algorithms(True)`.

**The GPU runs asynchronously.** A CUDA operation returns as soon as it is queued. Timing with `time.time()` around it measures the launch, not the work, unless you call `torch.cuda.synchronize()` first and after, or use CUDA events. Conversely, anything that needs a value on the host (`.item()`, `.cpu()`, `print(tensor)`, `if tensor > 0:`) waits for the GPU, so calling one every step stalls the pipeline.
