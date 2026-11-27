# Fill in a Triton softmax

A Triton kernel computes the softmax of each row of an `n_rows × n_cols` FP32 matrix, one program per row, with `BLOCK` a power of two ≥ `n_cols`. Fill in the blanks.
