#### The idea
`k <= 64` is tiny next to a 50,000-wide row, so almost every element can be rejected with one comparison. One block
of 64 threads per row:
1. Each thread scans a strided slice of the row and keeps its own top `k` in a small sorted array. A new value
   enters only if it beats the thread's current `k`-th best.
2. The 64 lists (at most 4,096 values) go to shared memory, are sorted there with a bitonic sort, and the first `k`
   are the answer. The global top `k` is always inside the union of the per-thread top `k`s, duplicates included.

#### A solution
```cuda
#define MAXK 64
#define THREADS 64

__global__ void topk_kernel(const float* X, float* Y, int cols, int k) {
    __shared__ float cand[THREADS * MAXK];                  // 16 KB
    const float* x = X + (size_t)blockIdx.x * cols;
    float best[MAXK];                                       // descending
    int n = 0;
    for (int c = threadIdx.x; c < cols; c += THREADS) {
        float v = x[c];
        if (n == k && v <= best[k - 1]) continue;           // the common case
        int p = n < k ? n++ : k - 1;                        // full: replace the smallest
        while (p > 0 && best[p - 1] < v) { best[p] = best[p - 1]; --p; }
        best[p] = v;
    }
    for (int i = 0; i < MAXK; ++i) cand[threadIdx.x * MAXK + i] = i < n ? best[i] : -INFINITY;
    __syncthreads();
    for (int size = 2; size <= THREADS * MAXK; size <<= 1)  // bitonic sort, descending
        for (int stride = size / 2; stride > 0; stride >>= 1) {
            for (int i = threadIdx.x; i < THREADS * MAXK; i += THREADS) {
                int j = i ^ stride;
                if (j > i) {
                    bool desc = (i & size) == 0;
                    float a = cand[i], b = cand[j];
                    if ((a < b) == desc) { cand[i] = b; cand[j] = a; }
                }
            }
            __syncthreads();
        }
    for (int i = threadIdx.x; i < k; i += THREADS) Y[(size_t)blockIdx.x * k + i] = cand[i];
}
// launch: rows blocks of THREADS threads
```
PyTorch: `Y.copy_(torch.topk(X, k, dim=-1).values)` (sorted by default).

#### What it costs
The scan reads each row once, `4 · cols` bytes, and does about one compare per element: memory-bound for long rows.
The bitonic step is `O(4096 · log² 4096)` compare-swaps per row, fixed and small. For large `k` the per-thread
arrays spill to local memory and a **radix select** wins: histogram the top bits of the values to find which bucket
holds the `k`-th largest, keep that bucket, repeat on the next bits.

#### Pitfalls
- Dropping duplicates. "The top 3 of [5, 5, 5, 1]" is [5, 5, 5].
- Sorting the whole row: correct, and tens of times more work than the task needs.
- A bitonic sort needs a power-of-two length; pad with `-INFINITY`.
