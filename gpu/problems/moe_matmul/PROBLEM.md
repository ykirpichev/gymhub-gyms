# MoE grouped matmul

**Level:** hard   **Teaches:** grouped GEMM, routing tokens by expert, gather plus tiled matmul, load imbalance

## Task

A mixture-of-experts layer has `E` expert weight matrices. A router has already picked one expert for each token. Multiply each token's row of `X` by its expert's matrix and write the result to the token's row of `Y`.

Do `matmul` first. This is the kernel at the heart of Mixtral, DeepSeek and other MoE models.

## Arguments

| Name     | Type           | Shape       | Role   |
| -------- | -------------- | ----------- | ------ |
| `X`      | `const float*` | `T × K`     | input  |
| `expert` | `const float*` | `T`         | input  |
| `W`      | `const float*` | `E × K × N` | input  |
| `Y`      | `float*`       | `T × N`     | output |
| `T`      | `int`          |             | size   |
| `E`      | `int`          |             | size   |
| `K`      | `int`          |             | size   |
| `N`      | `int`          |             | size   |

`expert` holds whole numbers in `[0, E)` stored as floats. All arrays are row-major: `W[e, k, n]` is at `W[(e * K + k) * N + n]`. An expert may get no tokens, and the routing in the benchmark is skewed: expert 0 gets about 60% of the tokens.

## Formula

```
e       = expert[t]
Y[t, n] = sum over k of X[t, k] * W[e, k, n]
```

## Example

```
X = [[1, 2],      expert = [1, 0, 1]      W[0] = [[1, 0],      W[1] = [[0, 1],
     [3, 4],                                      [0, 1]]              [1, 0]]
     [5, 6]]

token 0 uses W[1]: [2, 1]
token 1 uses W[0]: [3, 4]
token 2 uses W[1]: [6, 5]
Y = [[2, 1], [3, 4], [6, 5]]
```

## How to think about it

One thread per output, reading `expert[t]` and doing a dot product, is correct. It is also as slow as the naive `matmul`, and worse: neighbouring tokens use different experts, so there is no reuse of `W` at all.

A fast kernel turns this into `E` ordinary matmuls. First group the tokens by expert: count the tokens per expert, take a prefix sum of the counts to get each expert's start, and write a permutation that lists the tokens of expert 0, then expert 1, and so on. Then run a tiled matmul over the groups, where a tile of rows is a tile of the permutation: gather those rows of `X`, multiply by the expert's `W`, and scatter the results back to their tokens' rows of `Y`. A tile never mixes two experts.

Because the groups have different sizes, the number of row tiles per expert differs too. Launch one program per `(row tile, column tile)` over all groups, and let each program look up which expert and which rows it owns. That keeps the GPU busy even when one expert gets most of the tokens.

You may launch several kernels from `solve` and allocate scratch memory with `cudaMalloc` (or `torch.empty` in Triton and PyTorch).

## In an interview

Grouped GEMM for a mixture-of-experts layer: tokens of each expert, of different counts, multiplied by that expert's weights.

Follow-ups to answer out loud once your kernel passes:

- How do you group tokens by expert: sort, or a histogram and a scan for offsets?
- What happens to efficiency when one expert gets most of the tokens?
- How does expert parallelism move tokens between GPUs, and what does it cost?
