# Rotary position embedding

**Level:** medium   **Teaches:** frequency tables, broadcasting over batch and heads, the rotate_half layout

## Task

Apply rotary position embedding (RoPE) to a query (or key) tensor. Each head vector of size `Dh` is split into `Dh / 2` channel pairs, and each pair is rotated in its 2-D plane by an angle that grows with the token's position and shrinks with the pair's index. Token `t` of the input sits at absolute position `start + t` (during decoding, `start` is the number of tokens already in the cache). Return a new tensor.

Pairing convention ("rotate_half", as in Llama/GPT-NeoX): channel `i` pairs with channel `i + Dh/2`. It is not adjacent channels `(2i, 2i+1)`.

## Arguments

| Name    | Type           | Shape            | Role                                     |
| ------- | -------------- | ---------------- | ---------------------------------------- |
| `q`     | `torch.Tensor` | `[B, H, T, Dh]`  | input, float32, CUDA; `Dh` is even       |
| `start` | `int`          |                  | absolute position of token `t = 0`       |
| returns | `torch.Tensor` | `[B, H, T, Dh]`  | rotated output                           |

## Formula

```
half        = Dh / 2
inv_freq[i] = 10000 ^ (-2i / Dh)                 for i = 0 .. half-1
p           = start + t                          for t = 0 .. T-1
θ[t, i]     = p * inv_freq[i]

for every b, h, t and i < half, with x = q[b, h, t]:
  out[i]        = x[i]        * cos θ[t, i]  -  x[i + half] * sin θ[t, i]
  out[i + half] = x[i + half] * cos θ[t, i]  +  x[i]        * sin θ[t, i]
```

Equivalently, with `cos` and `sin` repeated to length `Dh` (`[cos θ, cos θ]`) and `rotate_half(x) = [-x[half:], x[:half]]`:

```
out = x * cos + rotate_half(x) * sin
```

## Example

```
Dh = 4, half = 2, start = 1, T = 1, q[0, 0, 0] = [1, 2, 3, 4]
inv_freq = [10000^0, 10000^(-1/2)] = [1, 0.01]
θ        = [1 * 1, 1 * 0.01]      = [1, 0.01]
pair (x0, x2) = (1, 3), angle 1:     out0 = 1·cos1 - 3·sin1 = -1.9841
                                     out2 = 3·cos1 + 1·sin1 =  2.4624
pair (x1, x3) = (2, 4), angle 0.01:  out1 = 2·cos.01 - 4·sin.01 = 1.9599
                                     out3 = 4·cos.01 + 2·sin.01 = 4.0198
out = [-1.9841, 1.9599, 2.4624, 4.0198]
```

## How to think about it

The angle table depends only on `(t, i)`: build a `[T, half]` table once and let it broadcast over `B` and `H`. Rotation preserves each pair's length, so the norm of every head vector is unchanged, which is a quick sanity check. Because a rotation by `p·ω` followed by a dot product only depends on the difference of positions, `q·k` after RoPE encodes relative position. Positions reach a few thousand, so expect float32 angle error around `1e-4`; the tolerance is `1e-3`.

## Tests

`Dh` from 2 to 128, positions starting at 0, 17, 1000 and 3000, odd `T`, and position 0 (where the output must equal the input). Benchmark input: `q[4, 32, 2048, 128]`, `start = 0`.
