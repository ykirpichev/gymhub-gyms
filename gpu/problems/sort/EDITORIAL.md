#### The idea
Comparison sorts on a GPU (bitonic, merge) do `O(N log² N)` or `O(N log N)` work. **Radix sort** does `O(N)` work per
pass and only a few passes, and it is what libraries use. For 32-bit keys, four passes of 8 bits, least significant
first; each pass is a stable counting sort:
1. Each block builds a 256-bin histogram of its keys' current digit.
2. A scan over all blocks' histograms gives every (digit, block) pair its output offset.
3. Each block scatters its keys to those offsets, keeping their order within a digit (stability is what makes LSD
   radix sort correct).

Floats sort correctly as unsigned integers after one bit flip: for a non-negative float set the sign bit; for a
negative one invert all bits. Then larger floats map to larger integers.

#### A solution
In production code you would call CUB, which ships with the CUDA toolkit and sorts `float` keys directly:

```cuda
#include <cub/cub.cuh>

extern "C" void solve(const float* A, float* B, int N) {
    void* tmp = nullptr;
    size_t bytes = 0;
    cub::DeviceRadixSort::SortKeys(tmp, bytes, A, B, N);   // first call: how much scratch?
    cudaMalloc(&tmp, bytes);
    cub::DeviceRadixSort::SortKeys(tmp, bytes, A, B, N);
    cudaFree(tmp);
    cudaDeviceSynchronize();
}
```
If you write it yourself, the key flip is the part to get exactly right:

```cuda
__device__ unsigned int to_key(float f) {
    unsigned int u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ float from_key(unsigned int k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}
```

#### What it costs
Each pass reads and writes every key once (plus small histograms): about `4 × 8N` bytes for four passes. At 16
million keys that is half a gigabyte of traffic, a few hundred microseconds on a fast GPU. A bitonic sort of the same
input does about 300 compare-exchange passes.

#### Pitfalls
- Sorting the raw float bits as integers: negative numbers come out reversed and after the positives.
- An unstable scatter within a pass: the earlier passes' order is lost and the result is wrong.
- Bitonic sort on a length that is not a power of two: pad with `+INFINITY` and drop the padding after.
