#### The idea
`B[i] = A[N - 1 - i]`. Each thread reads one element and writes one; the only question is whether the accesses
coalesce.

#### A solution
```cuda
__global__ void reverse_kernel(const float* A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) B[i] = A[N - 1 - i];
}
```

#### What it costs
Consecutive threads read consecutive addresses, just in descending order. A warp still touches the same 4 sectors
for its 128 bytes, so the loads coalesce fully: order within a warp does not matter, only which sectors are touched.
8 bytes per element, memory-bound.

If the task were *in place*, launch `N / 2` threads, each swapping `A[i]` and `A[N - 1 - i]`, so no thread reads
an element another thread has already overwritten.

#### Pitfalls
- An off-by-one: `A[N - i]` reads past the end for `i = 0`.
- Believing reversed access is uncoalesced and staging it through shared memory. Measure first: it gains nothing here.
