#### The idea
Each token uses a different weight matrix, so a plain matmul does not apply directly. Group tokens by expert, then
each group is an ordinary matmul with one weight matrix, and weights load once per tile instead of once per token.

1. Count tokens per expert (a histogram of `expert`).
2. Exclusive-scan the counts into offsets and write each token's index into its expert's slot (a counting sort).
3. Run a tiled matmul over each expert's group, gathering the rows of `X` through the permutation and scattering
   results back to `Y`.

#### A solution
The simple correct kernel, one thread per output, which every faster version must agree with:

```cuda
__global__ void moe_naive(const float* X, const float* expert, const float* W, float* Y, int T, int K, int N) {
    int t = blockIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float* w = W + (size_t)(int)expert[t] * K * N;
    const float* x = X + (size_t)t * K;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += x[k] * w[(size_t)k * N + n];    // coalesced across n
    Y[(size_t)t * N + n] = acc;
}
// grid (ceil(N / 128), T), 128 threads
```
The grouped version replaces the inner loop with your tiled matmul, where a block's tile of rows comes from one
expert's list of tokens. Its tile scheduler must handle uneven groups: expert 0 has 60 percent of the tokens here,
so give it proportionally more tiles rather than one block per expert.

#### What it costs
FLOPs are the same as a dense `T × K × N` matmul. The naive kernel reads a whole `K × N` weight matrix per token;
grouping reads it once per tile of tokens, which is what makes MoE affordable. With very few tokens per expert
(decoding), the kernel is memory-bound on the weights whatever you do.

#### Pitfalls
- Padding every expert to the largest group: correct, but wasteful under skewed routing.
- Forgetting the experts with no tokens: the scan must give them empty ranges, not garbage.
- `expert` holds floats; convert with `(int)`.
