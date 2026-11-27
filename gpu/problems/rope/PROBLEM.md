# Rotary embeddings (RoPE)

**Level:** medium   **Teaches:** 3D indexing, broadcasting a small table, reuse through the cache

## Task

Rotary position embeddings tell attention where each token sits by rotating its query and key vectors. Feature `i` of a head is paired with feature `i + D/2`, and each pair is rotated by an angle that depends on the token's position `t` and on `i`. The angles are precomputed: you get their cosines and sines as tables. Apply the rotation to every head of every token.

## Arguments

| Name  | Type           | Shape       | Role   |
| ----- | -------------- | ----------- | ------ |
| `X`   | `const float*` | `T × H × D` | input  |
| `cos` | `const float*` | `T × D/2`   | input  |
| `sin` | `const float*` | `T × D/2`   | input  |
| `Y`   | `float*`       | `T × H × D` | output |
| `T`   | `int`          |             | size   |
| `H`   | `int`          |             | size   |
| `D`   | `int`          |             | size   |

`T` is the number of tokens, `H` the number of heads and `D` the head size, which is even. Element `(t, h, d)` of `X` is at `X[(t * H + h) * D + d]`. Entry `(t, i)` of the tables is at `cos[t * (D / 2) + i]`. All heads of a token share the same angles.

## Formula

For every `t`, `h` and `i` in `[0, D/2)`, with `a = X[t, h, i]`, `b = X[t, h, i + D/2]`, `c = cos[t, i]` and `s = sin[t, i]`:

```
Y[t, h, i]       = a * c - b * s
Y[t, h, i + D/2] = b * c + a * s
```

This is the "rotate half" layout used by Llama and Hugging Face models. Some implementations pair neighbours `(2i, 2i + 1)` instead; this one does not.

## Example

```
T = 2, H = 1, D = 4
X   = [[[1, 2, 3, 4]],          cos = [[1,    1],     sin = [[0,    0],
       [[1, 2, 3, 4]]]                 [0.540, 1]]           [0.841, 0]]

token 0: no rotation           Y = [1, 2, 3, 4]
token 1, pair (1, 3):          1 * 0.540 - 3 * 0.841 = -1.984     3 * 0.540 + 1 * 0.841 = 2.462
token 1, pair (2, 4):          angle 0, unchanged
                               Y = [-1.984, 2, 2.462, 4]
```

## How to think about it

Every output depends on two inputs and two table entries, so the kernel is memory-bound and fully parallel. The natural unit of work is one pair: a thread that reads `a` and `b` writes both outputs, so nothing is read twice.

The tables are small next to `X`: `T × D/2` entries against `T × H × D`. All `H` heads of a token use the same row of the tables, so a launch that places the heads of one token next to each other reads each table row once from memory and then from cache. Count the bytes the benchmark reports and see how close you get.

## In an interview

Checks indexing on pairs of dimensions and whether you precompute or recompute the angles.

Follow-ups to answer out loud once your kernel passes:

- Interleaved pairs `(2i, 2i+1)` or split halves `(i, i + d/2)`? How do you know which a model uses?
- Compute `sin` and `cos` in the kernel or read them from a table? What does each cost?
- Fuse RoPE into the QKV projection or into attention. Which is easier, and why?

## This version and the other one

This is the kernel version: the angle tables are given, and you write the rotation in CUDA or Triton against a bandwidth benchmark. The LLM gym's `rope` is the PyTorch version: it builds the frequency table itself, handles a start offset for decoding, and returns a new tensor.
