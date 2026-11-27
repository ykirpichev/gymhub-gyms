# Make this softmax kernel fast

An interviewer shows you a CUDA softmax over the rows of a `rows × cols` FP32 matrix: one thread per row, looping over the row three times (max, sum of exponentials, write). It is slow for `rows = 4096, cols = 4096`. *"How would you make it fast? Go step by step."*
