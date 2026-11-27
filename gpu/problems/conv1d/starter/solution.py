import torch

# what is fullgraph?
# fullgraph is a flag that tells torch to compile the entire graph,
# including all the operations that are needed to compute the output.
# This is useful because it allows torch to optimize the graph for the best performance.
# However, it also means that the compiled function will be larger and slower to compile.
# So, we only use it when we need to optimize the performance of the function.
# In this case, we need to optimize the performance of the function,
# so we use fullgraph.
@torch.compile(fullgraph=True)
def solve(A: torch.Tensor, W: torch.Tensor, C: torch.Tensor, N: int, K: int) -> None:
    out_n = N - K + 1
    acc = A[:out_n] * W[0]
    for k in range(1, K):
        acc = acc + A[k:k + out_n] * W[k]
    C.copy_(acc)
